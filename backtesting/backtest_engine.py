"""Historical backtest engine for NSE data and strategy simulation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from backtesting.performance_metrics import PerformanceMetrics, PerformanceSummary
from backtesting.reports import BacktestReportWriter
from backtesting.transaction_costs import FillSimulator, TransactionCostModel
from backtesting.walk_forward import WalkForwardSplitter, run_walk_forward_validation
from features.feature_pipeline import FeaturePipeline
from market.data_loader import DataLoader
from ml.prediction_engine import PredictionEngine
from models import Mode, OrderSide, PredictionResult, StrategyType
from regime.regime_detector import RegimeDetector
from utils.config_loader import ConfigLoader
from utils.exceptions import DataValidationError


@dataclass(frozen=True)
class BacktestResult:
    """Backtest output bundle."""

    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    metrics: PerformanceSummary
    report_path: Path
    walk_forward: pd.DataFrame


@dataclass(frozen=True)
class BacktestConfig:
    """Backtest simulation parameters."""

    initial_capital: float = 1_000_000.0
    quantity: int = 75
    entry_buffer_percent: float = 0.0
    stop_loss_percent: float = 1.0
    target_percent: float = 1.5
    max_holding_bars: int = 5
    rolling_window: int = 20
    regime_lookback_bars: int = 60
    use_ml_prediction: bool = True
    run_walk_forward: bool = True
    dataset_name: str = "backtest"


class BacktestEngine:
    """Run deterministic backtests over historical OHLCV data."""

    def __init__(
        self,
        *,
        config_loader: ConfigLoader | None = None,
        feature_pipeline: FeaturePipeline | None = None,
        regime_detector: RegimeDetector | None = None,
        prediction_engine: PredictionEngine | None = None,
        data_loader: DataLoader | None = None,
        cost_model: TransactionCostModel | None = None,
        report_writer: BacktestReportWriter | None = None,
    ) -> None:
        """Create backtest engine."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        self.feature_pipeline = feature_pipeline or FeaturePipeline(mode=Mode.BACKTEST)
        self.regime_detector = regime_detector or RegimeDetector(feature_pipeline=self.feature_pipeline)
        self.prediction_engine = prediction_engine or PredictionEngine(config_loader=self.config_loader)
        self.data_loader = data_loader or DataLoader(config_loader=self.config_loader)
        self.cost_model = cost_model or TransactionCostModel()
        self.fill_simulator = FillSimulator()
        self.metrics = PerformanceMetrics()
        self.report_writer = report_writer or BacktestReportWriter()

    def run(
        self,
        historical_data: pd.DataFrame | None = None,
        *,
        config: BacktestConfig | None = None,
        save_report: bool = True,
    ) -> BacktestResult:
        """Run a backtest from historical NSE data."""

        active_config = config or BacktestConfig(initial_capital=self.config.risk_limits.capital)
        data = self._load_or_prepare_data(historical_data)
        features = self.feature_pipeline.generate_features(
            data,
            dataset_name=f"{active_config.dataset_name}_features",
            rolling_window=active_config.rolling_window,
            cache=False,
        ).features
        if active_config.use_ml_prediction and hasattr(self.prediction_engine, "reset_summary"):
            self.prediction_engine.reset_summary()
        trades = self._simulate(features, active_config)
        if active_config.use_ml_prediction and hasattr(self.prediction_engine, "log_summary"):
            self.prediction_engine.log_summary()
        equity_curve = self._equity_curve(trades, features, initial_capital=active_config.initial_capital)
        metrics = self.metrics.calculate(trades, equity_curve, initial_capital=active_config.initial_capital)
        walk_forward = self.walk_forward_validate(features, active_config) if active_config.run_walk_forward else pd.DataFrame()
        report_path = (
            self.report_writer.save(
                trades=trades,
                equity_curve=equity_curve,
                metrics=metrics,
                name=f"{active_config.dataset_name}_report",
                walk_forward=walk_forward,
            )
            if save_report
            else Path("")
        )
        return BacktestResult(trades, equity_curve, metrics, report_path, walk_forward)

    def walk_forward_validate(self, features: pd.DataFrame, config: BacktestConfig | None = None) -> pd.DataFrame:
        """Run walk-forward validation over feature data."""

        active_config = config or BacktestConfig(initial_capital=self.config.risk_limits.capital)

        def runner(_train: pd.DataFrame, test: pd.DataFrame, _window: Any) -> dict[str, float]:
            trades = self._simulate(test.reset_index(drop=True), active_config)
            equity = self._equity_curve(trades, test, initial_capital=active_config.initial_capital)
            summary = self.metrics.calculate(trades, equity, initial_capital=active_config.initial_capital)
            return {
                "total_return": summary.total_return,
                "max_drawdown": summary.max_drawdown,
                "win_rate": summary.win_rate,
                "expectancy": summary.expectancy,
            }

        train_size = min(max(active_config.rolling_window * 4, 20), max(20, len(features) // 2))
        test_size = min(max(active_config.rolling_window, 10), max(10, len(features) // 4))
        return run_walk_forward_validation(
            features,
            runner,
            splitter=WalkForwardSplitter(train_size=train_size, test_size=test_size),
        )

    def _simulate(self, features: pd.DataFrame, config: BacktestConfig) -> pd.DataFrame:
        trades: list[dict[str, Any]] = []
        minimum_index = max(config.rolling_window, 2)
        index = minimum_index
        while index < len(features) - 1:
            current = features.iloc[index]
            direction = self._direction(current)
            if direction == "flat":
                index += 1
                continue
            regime_start = max(0, index - config.regime_lookback_bars + 1)
            regime = self.regime_detector.detect(features.iloc[regime_start : index + 1], symbol="NIFTY")
            prediction = (
                self.prediction_engine.predict(
                    current.to_dict(),
                    regime,
                    timestamp=pd.to_datetime(current["date"]).to_pydatetime(),
                )
                if config.use_ml_prediction
                else None
            )
            if prediction is not None and not prediction.trade_allowed:
                index += 1
                continue
            trade = self._simulate_trade(
                features,
                index,
                direction,
                regime.regime_name or regime.regime.value,
                config,
                prediction=prediction,
            )
            trades.append(trade)
            index = int(trade["exit_index"]) + 1
        return pd.DataFrame(trades)

    def _simulate_trade(
        self,
        features: pd.DataFrame,
        entry_index: int,
        direction: str,
        regime_name: str,
        config: BacktestConfig,
        prediction: PredictionResult | None = None,
    ) -> dict[str, Any]:
        entry_bar = features.iloc[entry_index]
        entry_side = OrderSide.BUY if direction == "long" else OrderSide.SELL
        entry_price = float(entry_bar["close"]) * (1.0 + config.entry_buffer_percent / 100.0 * (1 if direction == "long" else -1))
        entry_fill = self.fill_simulator.limit_fill(
            side=entry_side,
            limit_price=entry_price,
            high=float(entry_bar["high"]),
            low=float(entry_bar["low"]),
        )
        if not entry_fill.filled:
            return self._unfilled_trade(entry_bar, entry_index, direction, regime_name)

        stop_price = entry_price * (1.0 - config.stop_loss_percent / 100.0) if direction == "long" else entry_price * (1.0 + config.stop_loss_percent / 100.0)
        target_price = entry_price * (1.0 + config.target_percent / 100.0) if direction == "long" else entry_price * (1.0 - config.target_percent / 100.0)
        exit_index = min(entry_index + config.max_holding_bars, len(features) - 1)
        exit_price = float(features.iloc[exit_index]["close"])
        exit_reason = "expiry_payoff"
        for idx in range(entry_index + 1, exit_index + 1):
            bar = features.iloc[idx]
            high = float(bar["high"])
            low = float(bar["low"])
            if direction == "long" and low <= stop_price:
                sl_fill = self.fill_simulator.sl_limit_fill(side=OrderSide.SELL, trigger_price=stop_price, limit_price=stop_price, high=high, low=low)
                exit_index, exit_price, exit_reason = idx, sl_fill.fill_price or stop_price, "sl_limit"
                break
            if direction == "short" and high >= stop_price:
                sl_fill = self.fill_simulator.sl_limit_fill(side=OrderSide.BUY, trigger_price=stop_price, limit_price=stop_price, high=high, low=low)
                exit_index, exit_price, exit_reason = idx, sl_fill.fill_price or stop_price, "sl_limit"
                break
            if direction == "long" and high >= target_price:
                exit_index, exit_price, exit_reason = idx, target_price, "target"
                break
            if direction == "short" and low <= target_price:
                exit_index, exit_price, exit_reason = idx, target_price, "target"
                break
        signed_points = exit_price - entry_price if direction == "long" else entry_price - exit_price
        gross_pnl = signed_points * config.quantity
        costs = self.cost_model.estimate(price=entry_price, quantity=config.quantity, orders=2)
        net_pnl = gross_pnl - costs
        return {
            "entry_index": entry_index,
            "exit_index": exit_index,
            "entry_date": entry_bar["date"],
            "exit_date": features.iloc[exit_index]["date"],
            "direction": direction,
            "strategy_type": StrategyType.MOMENTUM_BREAKOUT.value if direction == "long" else StrategyType.MEAN_REVERSION.value,
            "regime": regime_name,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "quantity": config.quantity,
            "gross_pnl": gross_pnl,
            "transaction_cost": costs,
            "net_pnl": net_pnl,
            "exit_reason": exit_reason,
            "option_payoff_approximation": signed_points * config.quantity,
            **self._prediction_report_fields(config, prediction, entry_bar),
        }

    @staticmethod
    def _direction(row: pd.Series) -> str:
        value = float(row.get("rolling_mean_return", row.get("daily_return", 0.0)) or 0.0)
        if value > 0:
            return "long"
        if value < 0:
            return "short"
        return "flat"

    def _load_or_prepare_data(self, historical_data: pd.DataFrame | None) -> pd.DataFrame:
        if historical_data is not None:
            data = historical_data.copy()
        else:
            data, _report = self.data_loader.load_30_years_nse_data()
        required = {"date", "open", "high", "low", "close", "volume"}
        missing = required.difference(data.columns)
        if missing:
            raise DataValidationError(f"Backtest data missing required columns: {sorted(missing)}")
        data["date"] = pd.to_datetime(data["date"])
        return data.sort_values("date").reset_index(drop=True)

    @staticmethod
    def _equity_curve(trades: pd.DataFrame, features: pd.DataFrame, *, initial_capital: float) -> pd.DataFrame:
        dates = pd.to_datetime(features["date"]).reset_index(drop=True)
        equity = pd.DataFrame({"date": dates, "equity": initial_capital})
        if trades.empty:
            return equity
        running = initial_capital
        pnl_by_exit = trades.groupby("exit_index")["net_pnl"].sum().to_dict()
        values = []
        for idx in range(len(equity)):
            running += float(pnl_by_exit.get(idx, 0.0))
            values.append(running)
        equity["equity"] = values
        return equity

    @staticmethod
    def _unfilled_trade(entry_bar: pd.Series, entry_index: int, direction: str, regime_name: str) -> dict[str, Any]:
        return {
            "entry_index": entry_index,
            "exit_index": entry_index,
            "entry_date": entry_bar["date"],
            "exit_date": entry_bar["date"],
            "direction": direction,
            "strategy_type": "unfilled",
            "regime": regime_name,
            "entry_price": float(entry_bar["close"]),
            "exit_price": float(entry_bar["close"]),
            "quantity": 0,
            "gross_pnl": 0.0,
            "transaction_cost": 0.0,
            "net_pnl": 0.0,
            "exit_reason": "limit_not_filled",
            "option_payoff_approximation": 0.0,
            "prediction_source": "",
            "model_name": "",
            "model_version": "",
            "features_used_count": 0,
        }

    @staticmethod
    def _prediction_report_fields(
        config: BacktestConfig,
        prediction: PredictionResult | None,
        row: pd.Series,
    ) -> dict[str, Any]:
        if prediction is not None:
            return {
                "prediction_source": prediction.prediction_source,
                "model_name": prediction.model_name or "",
                "model_version": prediction.model_version or "",
                "features_used_count": prediction.features_used_count,
            }
        source = "SAMPLE_BASELINE" if "SAMPLE" in config.dataset_name.upper() else "FALLBACK_RULE_BASED"
        return {
            "prediction_source": source,
            "model_name": "",
            "model_version": "",
            "features_used_count": sum(1 for value in row.to_dict().values() if _is_numeric(value)),
        }


def _is_numeric(value: Any) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False

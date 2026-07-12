"""Research backtest engine V2 for quant strategy stack."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from backtesting.transaction_costs import TransactionCostModel
from features.feature_pipeline import FeaturePipeline
from market.data_loader import DataLoader
from models import Mode
from quant.distribution.expiry_distribution import ExpiryDistributionEngine
from quant.optimization.portfolio_risk_optimizer import PortfolioRiskOptimizer
from quant.stress.monte_carlo_stress import MonteCarloStressTester
from quant.volatility.garch_model import reset_garch_performance_stats, save_garch_performance_report
from quant.volatility.vol_forecaster import VolatilityForecaster
from regime.regime_detector import RegimeDetector
from reports.strategy_v2_report import StrategyV2ReportWriter
from strategies.strategy_selector_v2 import StrategySelectorV2
from utils.config_loader import ConfigLoader
from utils.exceptions import DataValidationError


@dataclass(frozen=True)
class BacktestV2Result:
    """Backtest V2 output bundle."""

    trades: pd.DataFrame
    report_path: Path
    summary: dict[str, Any]


@dataclass(frozen=True)
class BacktestV2Config:
    """Backtest V2 configuration."""

    initial_capital: float = 1_000_000.0
    lookback: int = 80
    step: int = 5
    days_to_expiry: int = 7
    max_rows: int | None = 500
    stress_test_enabled: bool = True
    use_sample_data: bool = False
    sample_rows: int = 220


@dataclass
class PortfolioState:
    """Mutable accounting state for the research simulation."""

    initial_capital: float
    current_equity: float
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    available_capital: float = 0.0
    capital_at_risk: float = 0.0
    peak_equity: float = 0.0
    drawdown: float = 0.0
    daily_realized_pnl: float = 0.0

    def __post_init__(self) -> None:
        self.available_capital = self.current_equity
        self.peak_equity = self.current_equity


@dataclass(frozen=True)
class ExpiryCleanupResult:
    """Positions retained and trades closed by one idempotent expiry pass."""

    active_positions: list[dict[str, Any]]
    closed_trades: list[dict[str, Any]]


class BacktestEngineV2:
    """Backtest the StrategySelectorV2 research stack without execution integration."""

    def __init__(
        self,
        *,
        config_loader: ConfigLoader | None = None,
        data_loader: DataLoader | None = None,
        report_dir: str | Path = "reports/backtests",
        debug_dir: str | Path = "reports/debug",
    ) -> None:
        """Create BacktestEngineV2."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        self.data_loader = data_loader or DataLoader(config_loader=self.config_loader)
        self.feature_pipeline = FeaturePipeline(mode=Mode.BACKTEST)
        self.regime_detector = RegimeDetector(feature_pipeline=self.feature_pipeline)
        self.vol_forecaster = VolatilityForecaster()
        self.distribution_engine = ExpiryDistributionEngine(monte_carlo_simulations=1000)
        self.selector = StrategySelectorV2()
        self.optimizer = PortfolioRiskOptimizer(max_var_percent=self.config.risk_limits.max_daily_loss_percent)
        self.stress = MonteCarloStressTester()
        self.cost_model = TransactionCostModel()
        self.report_writer = StrategyV2ReportWriter()
        self.report_dir = Path(report_dir)
        self.debug_dir = Path(debug_dir)

    def run(self, historical_data: pd.DataFrame | None = None, *, config: BacktestV2Config | None = None) -> BacktestV2Result:
        """Run V2 research backtest and generate daily entry-audit diagnostics."""

        reset_garch_performance_stats()
        active = config or BacktestV2Config(initial_capital=self.config.risk_limits.capital)
        data, data_source, source_files = self._load_data(historical_data, active)
        source_rows = len(data)
        if active.max_rows:
            data = data.tail(active.max_rows).reset_index(drop=True)
        features = self.feature_pipeline.generate_features(data, cache=False).features
        trades: list[dict[str, Any]] = []
        candidate_rows: list[dict[str, Any]] = []
        rejected_rows: list[dict[str, Any]] = []
        distribution_rows: list[dict[str, Any]] = []
        vol_rows: list[dict[str, Any]] = []
        stress_rows: list[dict[str, Any]] = []
        no_trade_rows: list[dict[str, Any]] = []
        daily_signal_rows: list[dict[str, Any]] = []
        open_positions: list[dict[str, Any]] = []
        portfolio = PortfolioState(active.initial_capital, active.initial_capital)

        for index in range(len(features)):
            current = features.iloc[index]
            cleanup = _prune_expired_positions(index, open_positions, current, portfolio, self.cost_model)
            open_positions = cleanup.active_positions
            trades.extend(cleanup.closed_trades)
            scheduled_for_execution = index >= active.lookback and (index - active.lookback) % max(1, active.step) == 0
            if index < active.lookback:
                daily_signal_rows.append(
                    _basic_daily_row(
                        index=index,
                        current=current,
                        final_decision="REJECTED_INSUFFICIENT_LOOKBACK",
                        rejection_reason=f"Need {active.lookback} lookback rows before evaluating V2 filters.",
                    )
                )
                continue
            if index + active.days_to_expiry >= len(features):
                daily_signal_rows.append(
                    _basic_daily_row(
                        index=index,
                        current=current,
                        final_decision="REJECTED_INSUFFICIENT_FORWARD_EXIT_DATA",
                        rejection_reason=f"Need {active.days_to_expiry} future bars only for payoff simulation.",
                    )
                )
                continue

            window = features.iloc[index - active.lookback : index + 1].reset_index(drop=True)
            future = features.iloc[index + active.days_to_expiry]
            returns = window["close"].pct_change().dropna().to_numpy()
            regime = self.regime_detector.detect(window, symbol="NIFTY")
            vol = self.vol_forecaster.ensemble_forecast(returns, days_to_expiry=active.days_to_expiry)
            distribution = self.distribution_engine.estimate(
                spot_price=float(current["close"]),
                historical_returns=returns,
                days_to_expiry=active.days_to_expiry,
                realized_volatility=vol.forecast_vol,
                predicted_volatility=vol.forecast_vol,
                regime_result=regime,
                news_event_risk={"event_risk_level": "LOW"},
            )
            distribution_rows.append(_distribution_row(index, current, distribution))
            vol_rows.append(_vol_row(index, current, vol))
            option_chain = _synthetic_option_chain(
                float(current["close"]),
                active.days_to_expiry,
                as_of=pd.to_datetime(current["date"]).to_pydatetime(),
            )
            liquidity_score = float(current.get("liquidity_score", 0.75) or 0.75)
            momentum_score = float(current.get("rolling_mean_return", current.get("daily_return", 0.0)) or 0.0)
            prediction = SimpleNamespace(
                metadata={
                    "iv_percentile": 65.0,
                    "liquidity_score": liquidity_score,
                    "implied_volatility": max(vol.forecast_vol * 1.25, 0.20),
                    "realized_volatility": float(current.get("realized_volatility", vol.forecast_vol) or vol.forecast_vol),
                },
                breakout_failure_probability=0.65,
            )
            risk_limits = {
                "capital": portfolio.current_equity,
                "max_loss_per_trade_percent": self.config.risk_limits.max_loss_per_trade_percent,
            }
            selection = self.selector.select(
                current_regime=regime,
                prediction_result=prediction,
                expiry_distribution=distribution,
                volatility_forecast=vol,
                option_chain=option_chain,
                news_risk={"event_risk_level": "LOW"},
                risk_limits=risk_limits,
            )
            for row in selection.all_candidates_summary:
                candidate_rows.append({"index": index, "date": current.get("date", index), **row})
                if row.get("reject_reason"):
                    rejected_rows.append({"index": index, "date": current.get("date", index), **row})

            candidate = selection.selected_candidate
            size = None
            stress = None
            final_decision = "REJECTED_NO_CANDIDATE"
            rejection_reason = selection.explanation
            if candidate is not None:
                size = self.optimizer.recommend_size(
                    candidate_strategy=candidate,
                    capital=portfolio.current_equity,
                    max_risk_per_trade=portfolio.current_equity * self.config.risk_limits.max_loss_per_trade_percent / 100.0,
                    max_daily_risk=portfolio.current_equity * self.config.risk_limits.max_daily_loss_percent / 100.0,
                    current_open_positions=open_positions,
                    var=0.0,
                    cvar=0.0,
                    news_risk={"event_risk_level": "LOW"},
                )
                if portfolio.current_equity <= 0:
                    final_decision = "REJECTED_CAPITAL_EXHAUSTED"
                    rejection_reason = "Current equity is zero or negative."
                elif size.quantity < 1:
                    final_decision = "REJECTED_POSITION_SIZING"
                    rejection_reason = size.reason
                else:
                    stress = self.stress.run(
                        selected_strategy_candidate=candidate,
                        expiry_distribution=distribution,
                        max_loss_limit=size.risk_amount,
                    ) if active.stress_test_enabled else None
                    if stress is not None and stress.pass_fail == "FAIL":
                        final_decision = "REJECTED_RISK_MANAGEMENT"
                        rejection_reason = stress.explanation
                    elif not scheduled_for_execution:
                        final_decision = "REJECTED_BY_EVALUATION_SCHEDULE"
                        rejection_reason = f"V2 step={active.step} evaluates execution only every {active.step} bars."
                    else:
                        final_decision = "EXECUTED"
                        rejection_reason = "Passed V2 filters and execution schedule."

            if stress is not None:
                stress_rows.extend({"index": index, "date": current.get("date", index), **row} for row in stress.scenario_table)

            daily_row = _daily_signal_row(
                index=index,
                current=current,
                regime=regime,
                prediction=prediction,
                distribution=distribution,
                vol=vol,
                selection=selection,
                candidate=candidate,
                size=size,
                stress=stress,
                risk_limits=risk_limits,
                open_positions=open_positions,
                scheduled_for_execution=scheduled_for_execution,
                final_decision=final_decision,
                rejection_reason=rejection_reason,
                momentum_score=momentum_score,
                liquidity_score=liquidity_score,
            )
            daily_signal_rows.append(daily_row)

            if final_decision != "EXECUTED":
                no_trade_rows.append(
                    {
                        "index": index,
                        "date": current.get("date", index),
                        "reason": rejection_reason,
                        "rejection_count": len(selection.rejection_reasons),
                        "final_decision": final_decision,
                    }
                )
                continue

            assert candidate is not None and size is not None
            open_positions.append(
                {
                    "date": current.get("date", index),
                    "entry_index": index,
                    "exit_index": index + active.days_to_expiry,
                    "strategy": selection.selected_strategy,
                    "regime": regime.regime_name or regime.regime.value,
                    "volatility_state": vol.volatility_state,
                    "news_risk": "LOW",
                    "days_to_expiry": active.days_to_expiry,
                    "quantity": size.quantity,
                    "candidate": candidate,
                    "expiry_spot": float(future["close"]),
                    "risk_amount": size.risk_amount,
                    "closed": False,
                    "stress_pass_fail": stress.pass_fail if stress else "SKIPPED",
                }
            )

        if len(features):
            cleanup = _prune_expired_positions(len(features) - 1, open_positions, features.iloc[-1], portfolio, self.cost_model, force=True)
            open_positions = cleanup.active_positions
            trades.extend(cleanup.closed_trades)

        trades_frame = pd.DataFrame(trades)
        daily_signal_log = pd.DataFrame(daily_signal_rows)
        debug_paths = self._save_debug_reports(daily_signal_log)
        garch_report = save_garch_performance_report(self.debug_dir / "garch_performance_report.xlsx")
        summary = {
            "trades": len(trades_frame),
            "net_pnl": float(trades_frame["net_pnl"].sum()) if not trades_frame.empty else 0.0,
            "gross_pnl": float(trades_frame["gross_pnl"].sum()) if not trades_frame.empty else 0.0,
            "transaction_cost": float(trades_frame["total_transaction_cost"].sum()) if not trades_frame.empty else 0.0,
            "no_trade_count": len(no_trade_rows),
            "source_rows": source_rows,
            "rows_tested": len(features),
            "data_source": data_source,
            "data_rows": source_rows,
            "first_date": str(pd.to_datetime(data["date"]).min().date()),
            "last_date": str(pd.to_datetime(data["date"]).max().date()),
            "source_files": ", ".join(source_files),
            "ending_equity": portfolio.current_equity,
            "peak_equity": portfolio.peak_equity,
            "drawdown": portfolio.drawdown,
            "debug_reports": {**debug_paths, "garch_performance_report": str(garch_report)},
        }
        report_path = self._save_report(
            trades_frame,
            summary=summary,
            strategy_candidates=pd.DataFrame(candidate_rows),
            rejected_candidates=pd.DataFrame(rejected_rows),
            expiry_distribution=pd.DataFrame(distribution_rows),
            vol_forecasts=pd.DataFrame(vol_rows),
            stress_tests=pd.DataFrame(stress_rows),
            no_trade_days=pd.DataFrame(no_trade_rows),
        )
        summary["report_path"] = str(report_path)
        return BacktestV2Result(trades_frame, report_path, summary)

    def _load_data(self, historical_data: pd.DataFrame | None, config: BacktestV2Config) -> tuple[pd.DataFrame, str, list[str]]:
        if historical_data is not None:
            return historical_data.reset_index(drop=True), "REAL", ["provided_dataframe"]
        if config.use_sample_data:
            return _sample_data(config.sample_rows), "SAMPLE", ["deterministic_generated_sample"]
        data, report = self.data_loader.load_30_years_nse_data(dataset_name="backtest_v2_clean")
        files = [str(path) for path in self.data_loader._discover_files()]
        return data.reset_index(drop=True), "REAL", files

    def _save_debug_reports(self, daily_signal_log: pd.DataFrame) -> dict[str, str]:
        self.debug_dir.mkdir(parents=True, exist_ok=True)
        daily_path = self.debug_dir / "daily_signal_log.parquet"
        daily_signal_log.to_parquet(daily_path, index=False)
        stats = _filter_statistics(daily_signal_log)
        bottleneck = stats.sort_values("rejected_percent", ascending=False).head(1)
        summary_path = self.debug_dir / "trade_rejection_summary.xlsx"
        stats_path = self.debug_dir / "trade_filter_statistics.xlsx"
        with pd.ExcelWriter(summary_path) as writer:
            pd.DataFrame(
                [
                    {
                        "total_days": len(daily_signal_log),
                        "executed_trades": int((daily_signal_log["Final decision"] == "EXECUTED").sum()) if not daily_signal_log.empty else 0,
                        "rejected_days": int((daily_signal_log["Final decision"] != "EXECUTED").sum()) if not daily_signal_log.empty else 0,
                        "highest_rejection_filter": bottleneck.iloc[0]["filter"] if not bottleneck.empty else "none",
                        "highest_rejection_percent": bottleneck.iloc[0]["rejected_percent"] if not bottleneck.empty else 0.0,
                    }
                ]
            ).to_excel(writer, sheet_name="Summary", index=False)
            stats.to_excel(writer, sheet_name="Rejection Counts", index=False)
            _safe_sheet(daily_signal_log.head(5000)).to_excel(writer, sheet_name="Daily Audit Sample", index=False)
        with pd.ExcelWriter(stats_path) as writer:
            stats.to_excel(writer, sheet_name="Filter Statistics", index=False)
            _decision_stats(daily_signal_log).to_excel(writer, sheet_name="Final Decisions", index=False)
            _regime_stats(daily_signal_log).to_excel(writer, sheet_name="By Regime", index=False)
        return {
            "trade_rejection_summary": str(summary_path),
            "trade_filter_statistics": str(stats_path),
            "daily_signal_log": str(daily_path),
        }

    def _save_report(
        self,
        trades: pd.DataFrame,
        *,
        summary: dict[str, Any],
        strategy_candidates: pd.DataFrame,
        rejected_candidates: pd.DataFrame,
        expiry_distribution: pd.DataFrame,
        vol_forecasts: pd.DataFrame,
        stress_tests: pd.DataFrame,
        no_trade_days: pd.DataFrame,
    ) -> Path:
        return self.report_writer.save(
            report_dir=self.report_dir,
            summary=summary,
            trades=trades,
            strategy_candidates=strategy_candidates,
            rejected_candidates=rejected_candidates,
            expiry_distribution=expiry_distribution,
            vol_forecasts=vol_forecasts,
            stress_tests=stress_tests,
            no_trade_days=no_trade_days,
        )


def _payoff_at(candidate: Any, expiry_price: float) -> float:
    curve = list(getattr(candidate, "payoff_curve", ()) or ())
    if not curve:
        return float(getattr(candidate, "expected_value", 0.0) or 0.0)
    nearest = min(curve, key=lambda row: abs(float(row["expiry_price"]) - expiry_price))
    return float(nearest["payoff"])


def _prune_expired_positions(
    current_index: int,
    active_positions: list[dict[str, Any]],
    current_market_data: pd.Series,
    portfolio_state: PortfolioState,
    cost_model: TransactionCostModel,
    *,
    force: bool = False,
) -> ExpiryCleanupResult:
    """Close matured positions exactly once and update portfolio accounting."""

    retained: list[dict[str, Any]] = []
    closed: list[dict[str, Any]] = []
    portfolio_state.daily_realized_pnl = 0.0
    for position in active_positions:
        if position.get("closed"):
            continue
        if not force and int(position["exit_index"]) > current_index:
            retained.append(position)
            continue
        candidate = position["candidate"]
        expiry_spot = float(position.get("expiry_spot", current_market_data["close"]))
        multiplier = int(position["quantity"])
        gross = _payoff_at(candidate, expiry_spot) * multiplier
        costs = _candidate_costs(candidate, expiry_spot, multiplier, cost_model)
        net = gross - costs["total_transaction_cost"]
        trade = {key: value for key, value in position.items() if key not in {"candidate", "closed", "expiry_spot", "risk_amount"}}
        trade.update(costs)
        trade.update(
            {
                "exit_date": current_market_data.get("date", current_index),
                "premium_source": "APPROXIMATED",
                "gross_pnl": gross,
                "transaction_cost": costs["brokerage"] + costs["statutory_charges"],
                "slippage": costs["slippage_cost"],
                "net_pnl": net,
            }
        )
        position["closed"] = True
        closed.append(trade)
        portfolio_state.realized_pnl += net
        portfolio_state.daily_realized_pnl += net
    portfolio_state.capital_at_risk = sum(float(row.get("risk_amount", 0.0)) for row in retained)
    portfolio_state.current_equity = portfolio_state.initial_capital + portfolio_state.realized_pnl + portfolio_state.unrealized_pnl
    portfolio_state.available_capital = portfolio_state.current_equity - portfolio_state.capital_at_risk
    portfolio_state.peak_equity = max(portfolio_state.peak_equity, portfolio_state.current_equity)
    portfolio_state.drawdown = portfolio_state.current_equity - portfolio_state.peak_equity
    return ExpiryCleanupResult(retained, closed)


def _candidate_costs(candidate: Any, expiry_spot: float, size_multiplier: int, cost_model: TransactionCostModel) -> dict[str, float]:
    totals = {key: 0.0 for key in ("entry_turnover", "exit_turnover", "total_turnover", "brokerage", "statutory_charges", "slippage_cost", "total_transaction_cost")}
    for leg in getattr(candidate, "legs", ()):
        option_type = str(leg.get("option_type", "CE")).upper()
        strike = float(leg.get("strike", 0.0))
        exit_price = max(expiry_spot - strike, 0.0) if option_type == "CE" else max(strike - expiry_spot, 0.0)
        breakdown = cost_model.estimate_round_trip(
            entry_price=float(leg.get("premium", 0.0)),
            exit_price=exit_price,
            quantity=max(int(leg.get("quantity", 1)), 1) * max(size_multiplier, 1),
            lot_size=max(int(leg.get("lot_size", 1)), 1),
        )
        for key in totals:
            totals[key] += breakdown[key]
    return totals


def _synthetic_option_chain(spot: float, days_to_expiry: int, *, as_of: datetime):
    from models import OptionChainSnapshot, OptionContract

    step = 100.0
    base = round(spot / step) * step
    snapshot_time = as_of if isinstance(as_of, datetime) else datetime.combine(as_of, time.min)
    expiry = snapshot_time.date() + timedelta(days=days_to_expiry)
    contracts = []
    for offset in range(-5, 6):
        strike = base + offset * step
        distance = abs(strike - spot)
        call = max(8.0, (spot - strike) * 0.4 + 120.0 - distance * 0.25)
        put = max(8.0, (strike - spot) * 0.4 + 120.0 - distance * 0.25)
        contracts.append(OptionContract(f"NIFTY{int(strike)}CE", "NIFTY", expiry, strike, "CE", 75, bid=call - 0.5, ask=call + 0.5, last_price=call))
        contracts.append(OptionContract(f"NIFTY{int(strike)}PE", "NIFTY", expiry, strike, "PE", 75, bid=put - 0.5, ask=put + 0.5, last_price=put))
    return OptionChainSnapshot("NIFTY", snapshot_time, spot, expiry, tuple(contracts), metadata={"source": "synthetic_backtest"})


def _basic_daily_row(*, index: int, current: pd.Series, final_decision: str, rejection_reason: str) -> dict[str, Any]:
    return {
        "Date": current.get("date", index),
        "Index": index,
        "Regime": "NOT_EVALUATED",
        "ML probability": 0.0,
        "Expected move": 0.0,
        "News risk": "LOW",
        "Volatility forecast": 0.0,
        "Liquidity score": float(current.get("liquidity_score", 0.0) or 0.0),
        "Trend score": 0.0,
        "Momentum score": float(current.get("rolling_mean_return", current.get("daily_return", 0.0)) or 0.0),
        "Option premium score": 0.0,
        "Risk limits": "not evaluated",
        "Portfolio exposure": 0.0,
        "Position sizing": "not evaluated",
        "Entry signal": "NO_SIGNAL",
        "Final decision": final_decision,
        "Rejection reason": rejection_reason,
        **_rejection_flags(final_decision, rejection_reason),
    }


def _daily_signal_row(**kwargs: Any) -> dict[str, Any]:
    current = kwargs["current"]
    regime = kwargs["regime"]
    prediction = kwargs["prediction"]
    distribution = kwargs["distribution"]
    vol = kwargs["vol"]
    selection = kwargs["selection"]
    candidate = kwargs["candidate"]
    size = kwargs["size"]
    stress = kwargs["stress"]
    risk_limits = kwargs["risk_limits"]
    open_positions = kwargs["open_positions"]
    final_decision = kwargs["final_decision"]
    rejection_reason = kwargs["rejection_reason"]
    max_loss = float(getattr(candidate, "max_loss", 0.0) or 0.0) if candidate is not None else 0.0
    option_premium_score = _option_premium_score(candidate)
    sizing_text = "not evaluated" if size is None else f"qty={size.quantity}; risk={size.risk_amount:.2f}; {size.reason}"
    stress_text = "SKIPPED" if stress is None else stress.pass_fail
    return {
        "Date": current.get("date", kwargs["index"]),
        "Index": kwargs["index"],
        "Regime": regime.regime_name or regime.regime.value,
        "ML probability": float(getattr(prediction, "breakout_failure_probability", 0.0) or 0.0),
        "Expected move": float(distribution.expected_move_percent),
        "News risk": "LOW",
        "Volatility forecast": float(vol.forecast_vol),
        "Liquidity score": kwargs["liquidity_score"],
        "Trend score": float(getattr(regime, "trend_score", 0.0) or 0.0),
        "Momentum score": kwargs["momentum_score"],
        "Option premium score": option_premium_score,
        "Risk limits": f"capital={risk_limits['capital']}; max_loss_per_trade_percent={risk_limits['max_loss_per_trade_percent']}",
        "Portfolio exposure": float(sum(float(position.get("risk_amount", 0.0) or 0.0) for position in open_positions)),
        "Position sizing": sizing_text,
        "Entry signal": selection.selected_strategy,
        "Final decision": final_decision,
        "Rejection reason": rejection_reason,
        "Candidate max loss": max_loss,
        "Candidate EV": float(getattr(candidate, "expected_value", 0.0) or 0.0) if candidate is not None else 0.0,
        "Candidate POP": float(getattr(candidate, "probability_of_profit", 0.0) or 0.0) if candidate is not None else 0.0,
        "Stress result": stress_text,
        "Scheduled for execution": bool(kwargs["scheduled_for_execution"]),
        **_rejection_flags(final_decision, rejection_reason, selection=selection),
    }


def _distribution_row(index: int, current: pd.Series, distribution: Any) -> dict[str, Any]:
    return {
        "index": index,
        "date": current.get("date", index),
        "expected_expiry_price": distribution.expected_expiry_price,
        "expected_move_points": distribution.expected_move_points,
        "expected_move_percent": distribution.expected_move_percent,
        "lower_bound_68": distribution.lower_bound_68,
        "upper_bound_68": distribution.upper_bound_68,
        "lower_bound_95": distribution.lower_bound_95,
        "upper_bound_95": distribution.upper_bound_95,
        "tail_risk_score": distribution.tail_risk_score,
        "model_used": distribution.model_used,
    }


def _vol_row(index: int, current: pd.Series, vol: Any) -> dict[str, Any]:
    return {
        "index": index,
        "date": current.get("date", index),
        "forecast_vol": vol.forecast_vol,
        "forecast_vol_percent": vol.forecast_vol_percent,
        "volatility_state": vol.volatility_state,
        "confidence": vol.confidence,
        "model_used": vol.model_used,
        "warning": vol.warning,
    }


def _option_premium_score(candidate: Any) -> float:
    if candidate is None:
        return 0.0
    max_loss = float(getattr(candidate, "max_loss", 0.0) or 0.0)
    max_profit = float(getattr(candidate, "max_profit", 0.0) or 0.0)
    return max_profit / max(max_loss, 1.0)


def _rejection_flags(final_decision: str, reason: str, *, selection: Any | None = None) -> dict[str, bool]:
    text = f"{final_decision} {reason} " + " ".join(getattr(selection, "rejection_reasons", ()) or ())
    lower = text.lower()
    return {
        "Rejected by news": "news" in lower,
        "Rejected by regime": "regime" in lower or "trend" in lower,
        "Rejected by ML confidence": "confidence" in lower or "ml" in lower,
        "Rejected by expected move": "expected move" in lower,
        "Rejected by volatility": "volatility" in lower or "iv percentile" in lower,
        "Rejected by probability threshold": "probability" in lower or "pop" in lower,
        "Rejected by risk management": "risk" in lower or "stress" in lower or "max loss" in lower or "var" in lower or "cvar" in lower,
        "Rejected by position sizing": "position" in lower or "quantity < 1" in lower or final_decision == "REJECTED_POSITION_SIZING",
        "Rejected by evaluation schedule": final_decision == "REJECTED_BY_EVALUATION_SCHEDULE",
        "Rejected by insufficient data": "INSUFFICIENT" in final_decision,
    }


def _filter_statistics(daily_signal_log: pd.DataFrame) -> pd.DataFrame:
    filters = [
        "Rejected by news",
        "Rejected by regime",
        "Rejected by ML confidence",
        "Rejected by expected move",
        "Rejected by volatility",
        "Rejected by probability threshold",
        "Rejected by risk management",
        "Rejected by position sizing",
        "Rejected by evaluation schedule",
        "Rejected by insufficient data",
    ]
    total = max(len(daily_signal_log), 1)
    rows = []
    for column in filters:
        count = int(daily_signal_log.get(column, pd.Series(dtype=bool)).fillna(False).sum())
        rows.append({"filter": column, "rejected_count": count, "rejected_percent": count / total * 100.0})
    return pd.DataFrame(rows).sort_values("rejected_percent", ascending=False).reset_index(drop=True)


def _decision_stats(daily_signal_log: pd.DataFrame) -> pd.DataFrame:
    if daily_signal_log.empty:
        return pd.DataFrame({"Final decision": [], "count": []})
    return daily_signal_log.groupby("Final decision", dropna=False).size().reset_index(name="count")


def _regime_stats(daily_signal_log: pd.DataFrame) -> pd.DataFrame:
    if daily_signal_log.empty:
        return pd.DataFrame({"Regime": [], "days": [], "executed": []})
    return daily_signal_log.groupby("Regime", dropna=False).agg(
        days=("Regime", "size"),
        executed=("Final decision", lambda values: int((values == "EXECUTED").sum())),
    ).reset_index()


def _safe_sheet(dataframe: pd.DataFrame) -> pd.DataFrame:
    return dataframe if not dataframe.empty else pd.DataFrame({"status": ["no rows"]})


def _sample_data(rows: int = 220) -> pd.DataFrame:
    rng = np.random.default_rng(17)
    returns = rng.normal(0.0002, 0.01, rows)
    close = 26000.0 * np.cumprod(1.0 + returns)
    high = close * 1.005
    low = close * 0.995
    return pd.DataFrame({"date": pd.date_range("2024-01-01", periods=rows, freq="B"), "open": close, "high": high, "low": low, "close": close, "volume": 100000})

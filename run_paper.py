"""Run a safe PAPER-mode smoke workflow."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from broker.kotak_client import KotakClient
from market.market_data_manager import MarketDataManager
from models import Mode
from news import NewsScheduler
from risk import RiskManager
from utils.config_loader import ConfigLoader
from utils.file_utils import ensure_parent_dir
from utils.logger import setup_logging
from utils.time_utils import now_ist

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run PAPER mode end-to-end using safe mocks and write dashboard state."""

    parser = argparse.ArgumentParser(description="Run PAPER mode smoke workflow")
    parser.add_argument("--state-dir", default="data/live", help="Dashboard state output directory")
    args = parser.parse_args(argv)

    loader = ConfigLoader()
    config = loader.load()
    setup_logging(level=config.settings.log_level)
    if config.settings.mode == Mode.LIVE or config.settings.live_trading_enabled:
        print("PAPER runner blocked because LIVE mode or live trading is enabled.")
        return 1

    broker = KotakClient(mode=Mode.PAPER, config_loader=loader)
    broker.login()
    broker.validate_session()
    market = MarketDataManager(mode=Mode.PAPER, config_loader=loader, broker_client=broker)
    spot = market.get_latest_spot("NIFTY")
    news_scheduler = NewsScheduler()
    news_risk = news_scheduler.refresh_if_due()
    margin = broker.get_margin()
    risk_manager = RiskManager(config_loader=loader)
    kill_state = risk_manager.kill_switch.state()
    state_dir = Path(args.state_dir)
    _write_json(state_dir / "regime.json", {"regime": "paper_mock", "regime_name": "PAPER mock regime"})
    _write_json(
        state_dir / "prediction.json",
        {
            "underlying": "NIFTY",
            "spot": spot,
            "trade_allowed": False,
            "reason": "PAPER smoke run only; no strategy executed.",
            "timestamp": now_ist().isoformat(),
        },
    )
    _write_json(
        state_dir / "risk.json",
        {
            "risk_used": 0.0,
            "available_margin": margin.available_margin,
            "kill_switch_active": kill_state.active,
        },
    )
    _write_json(state_dir / "kill_switch.json", {"active": kill_state.active, "reason": kill_state.reason})
    _write_json(
        state_dir / "news.json",
        {
            "latest_news": [article.headline for article in news_scheduler.latest_articles[:5]],
            "sentiment_score": news_scheduler.latest_sentiment.sentiment_score if news_scheduler.latest_sentiment else 0.0,
            "event_risk_level": news_risk.event_risk_level,
            "no_trade_warning": "" if news_risk.trade_allowed else news_risk.reason,
            "reason": news_risk.reason,
        },
    )
    _write_json(
        state_dir / "strategy_v2.json",
        {
            "selected_strategy": "NO_TRADE",
            "position_size": {"quantity": 0, "risk_amount": 0.0, "reason": "PAPER smoke run only."},
            "explanation": "Strategy V2 state generated for dashboard smoke workflow; no orders placed.",
            "timestamp": now_ist().isoformat(),
        },
    )
    _write_json(
        state_dir / "expiry_distribution.json",
        {
            "underlying": "NIFTY",
            "spot": spot,
            "expected_expiry_price": spot,
            "lower_bound_68": spot * 0.99,
            "upper_bound_68": spot * 1.01,
            "lower_bound_95": spot * 0.98,
            "upper_bound_95": spot * 1.02,
            "tail_risk_score": 0.0,
        },
    )
    _write_json(
        state_dir / "vol_forecast.json",
        {
            "forecast_vol": 0.0,
            "forecast_vol_percent": 0.0,
            "volatility_state": "paper_mock",
            "confidence": 0.0,
            "model_used": "paper_smoke",
        },
    )
    _write_json(
        state_dir / "strategy_candidates.json",
        {
            "candidates": [],
            "rejection_reasons": ["PAPER smoke run only; strategy research not executed."],
        },
    )
    _write_json(
        state_dir / "stress_test.json",
        {
            "pass_fail": "NOT_RUN",
            "worst_case_loss": 0.0,
            "probability_of_loss": 0.0,
            "explanation": "Stress test not run during PAPER smoke workflow.",
        },
    )
    _write_json(
        state_dir / "performance_attribution.json",
        {
            "net_pnl": 0.0,
            "cost_drag": 0.0,
            "gross_to_net_decay": 0.0,
            "status": "No trades in PAPER smoke workflow.",
        },
    )
    _write_csv(state_dir / "active_orders.csv", "order_id,symbol,status\n")
    _write_csv(state_dir / "open_positions.csv", "symbol,quantity,average_price,last_price,unrealized_pnl\n")
    LOGGER.info("PAPER smoke workflow completed", extra={"spot": spot})
    print(f"PAPER smoke complete. Spot={spot:.2f}. State written to {state_dir.resolve()}")
    return 0


def _write_json(path: Path, payload: dict) -> None:
    file_path = ensure_parent_dir(path)
    file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _write_csv(path: Path, content: str) -> None:
    file_path = ensure_parent_dir(path)
    file_path.write_text(content, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())

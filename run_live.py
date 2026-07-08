"""LIVE-mode preflight runner.

This module does not place orders. It verifies that live trading gates are
explicitly enabled and that required safety checks can pass before any future
live execution path is allowed.
"""

from __future__ import annotations

import argparse
import logging

from broker.kotak_client import KotakClient
from execution.order_validator import OrderValidator
from models import Mode
from news import NewsScheduler
from risk import RiskManager
from utils.config_loader import ConfigLoader
from utils.exceptions import BrokerError, ConfigError, ExecutionError
from utils.logger import setup_logging

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run LIVE safety preflight without placing orders."""

    parser = argparse.ArgumentParser(description="Run LIVE preflight checks")
    parser.add_argument("--dry-run", action="store_true", default=True, help="Always true: no live orders are placed")
    args = parser.parse_args(argv)
    _ = args

    try:
        loader = ConfigLoader()
        config = loader.load()
        setup_logging(level=config.settings.log_level)
        _validate_live_config(config.settings.mode, config.settings.live_trading_enabled)
        broker = KotakClient(mode=Mode.LIVE, config_loader=loader)
        if not broker.login() or not broker.validate_session():
            raise BrokerError("Valid broker session is required for LIVE mode")
        risk_manager = RiskManager(config_loader=loader)
        if risk_manager.kill_switch.active:
            raise ExecutionError("Kill switch must be inactive for LIVE mode")
        news_risk = NewsScheduler().refresh_if_due()
        if not news_risk.trade_allowed:
            raise ExecutionError(f"News event risk blocks LIVE mode: {news_risk.reason}")
        _ = OrderValidator(config_loader=loader)
    except (BrokerError, ConfigError, ExecutionError, ValueError) as exc:
        LOGGER.error("LIVE preflight blocked", extra={"reason": str(exc)})
        print(f"LIVE preflight blocked: {exc}")
        return 1

    print("LIVE preflight passed. No orders were placed.")
    return 0


def _validate_live_config(mode: Mode, live_trading_enabled: bool) -> None:
    if mode != Mode.LIVE:
        raise ExecutionError("LIVE mode requires config/settings.yaml mode: LIVE")
    if not live_trading_enabled:
        raise ExecutionError("LIVE mode requires live_trading_enabled: true")


if __name__ == "__main__":
    raise SystemExit(main())

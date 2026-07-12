"""Application entrypoint for safe startup and runner dispatch."""

from __future__ import annotations

import argparse
import logging

from models import Mode
from news import NewsConfigLoader
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging


def initialize_application() -> ConfigLoader:
    """Initialize logging and configuration without enabling live trading."""

    loader = ConfigLoader()
    config = loader.load()
    setup_logging(level=config.settings.log_level)
    logging.getLogger(__name__).info(
        "Application initialized",
        extra={
            "mode": config.settings.mode.value,
            "live_trading_enabled": config.settings.live_trading_enabled,
        },
    )
    try:
        news_config = NewsConfigLoader().load()
        logging.getLogger(__name__).info("News module initialized", extra={"enabled": news_config.enabled})
    except Exception as exc:
        logging.getLogger(__name__).warning("News module initialization skipped", extra={"error": type(exc).__name__})
    return loader


def main(argv: list[str] | None = None) -> int:
    """Initialize the app or dispatch to a safe runner."""

    parser = argparse.ArgumentParser(description="NIFTY50 trading system entrypoint")
    parser.add_argument(
        "command",
        nargs="?",
        choices=("init", "backtest", "paper", "live"),
        default="init",
        help="Action to run. Defaults to safe initialization only.",
    )
    args, forwarded = parser.parse_known_args(argv)
    loader = initialize_application()
    config = loader.load()
    if args.command == "init":
        print(f"Initialized trading system in {config.settings.mode.value} mode. Live enabled={config.settings.live_trading_enabled}.")
        return 0
    if args.command == "backtest":
        from run_backtest import main as run_backtest_main

        return run_backtest_main(forwarded)
    if forwarded:
        parser.error(f"unrecognized arguments: {' '.join(forwarded)}")
    if args.command == "paper":
        from run_paper import main as run_paper_main

        return run_paper_main([])
    if args.command == "live":
        if config.settings.mode != Mode.LIVE:
            print("LIVE runner blocked: config mode is not LIVE.")
            return 1
        from run_live import main as run_live_main

        return run_live_main([])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

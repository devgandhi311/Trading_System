from __future__ import annotations

import argparse
import logging
from pathlib import Path
import signal

from runtime.paper_trading_engine import PaperTradingEngine
from runtime.paper_qualification import PaperQualificationMode
from runtime.paper_market_input import PaperMarketInputProvider
from runtime.process_lock import PaperProcessLock, ProcessLockError
from market.market_data_manager import MarketDataManager
from models import Mode
from storage.database import OperationalDatabase
from utils.config_loader import ConfigLoader
from utils.logger import setup_logging

LOGGER = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> int:
    """Run the autonomous paper trading engine."""

    parser = argparse.ArgumentParser(description="Run the autonomous paper trading engine.")
    parser.add_argument("--once", action="store_true", help="Run one trading cycle and exit (default).")
    parser.add_argument("--continuous", action="store_true", help="Run cycles until shutdown.")
    parser.add_argument("--qualification", action="store_true", help="Run autonomous PAPER qualification mode.")
    parser.add_argument("--qualification-report-dir", type=Path, default=Path("reports/qualification"))
    parser.add_argument("--state-dir", type=Path, default=Path("data/live"), help="Dashboard projection directory.")
    parser.add_argument("--database", type=Path, default=None, help="Authoritative SQLite database path.")
    parser.add_argument("--poll-seconds", type=float, default=5.0, help="Continuous-cycle polling interval.")
    parser.add_argument("--demo-input", action="store_true", help="Explicitly allow demo PAPER input; prohibited in qualification.")
    args = parser.parse_args(argv)

    config_loader = ConfigLoader()
    setup_logging(level=config_loader.load().settings.log_level)

    database = OperationalDatabase(args.database or args.state_dir / "trading_state.db")
    manager = MarketDataManager(mode=Mode.PAPER, config_loader=config_loader)
    provider = PaperMarketInputProvider(manager, allow_demo=args.demo_input and not args.qualification)
    lock = PaperProcessLock(args.state_dir / "paper_runtime.lock")
    try:
        lock.acquire()
    except ProcessLockError:
        LOGGER.exception("PAPER process lock acquisition failed")
        return 1
    engine = None
    try:
        engine = PaperTradingEngine(config_loader, database=database, state_dir=args.state_dir,
                                    input_provider=provider, poll_interval_seconds=args.poll_seconds)
        for signal_name in (signal.SIGINT, signal.SIGTERM):
            signal.signal(signal_name, lambda _signum, _frame: engine.request_shutdown())
        if args.qualification:
            PaperQualificationMode(engine, report_dir=args.qualification_report_dir).run()
        else:
            engine.run(continuous=args.continuous)
    except Exception:
        LOGGER.exception("PAPER engine failed closed")
        return 1
    finally:
        if engine is not None:
            engine.close()
        lock.release()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

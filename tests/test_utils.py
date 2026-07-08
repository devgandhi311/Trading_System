from datetime import date, datetime, timezone
import logging

import pytest

from utils.exceptions import BrokerError, ConfigError, TradingSystemError
from utils.file_utils import save_csv, save_excel
from utils.logger import get_broker_logger, get_trade_logger, setup_logging
from utils.math_utils import calculate_pnl, clamp, percent_change, round_to_tick, safe_divide
from utils.time_utils import (
    KOLKATA_TZ,
    can_place_new_trade,
    ensure_ist,
    is_market_open,
    parse_hhmm,
    should_force_squareoff,
)


def test_custom_exceptions_share_base_type() -> None:
    assert issubclass(BrokerError, TradingSystemError)
    assert issubclass(ConfigError, TradingSystemError)
    assert issubclass(ConfigError, ValueError)


def test_time_helpers_are_timezone_and_market_aware() -> None:
    market_time = datetime(2026, 7, 8, 10, 0, tzinfo=KOLKATA_TZ)
    late_time = datetime(2026, 7, 8, 15, 25, tzinfo=KOLKATA_TZ)
    utc_time = datetime(2026, 7, 8, 4, 30, tzinfo=timezone.utc)

    assert ensure_ist(utc_time).tzinfo == KOLKATA_TZ
    assert parse_hhmm("09:15").hour == 9
    assert is_market_open(market_time)
    assert can_place_new_trade(market_time)
    assert should_force_squareoff(late_time)


def test_math_helpers() -> None:
    assert safe_divide(1, 0, default=-1) == -1
    assert clamp(10, 0, 5) == 5
    assert round_to_tick(100.03, 0.05) == 100.05
    assert percent_change(100, 110) == 10
    assert calculate_pnl(100, 110, 2, side="BUY") == 20
    assert calculate_pnl(100, 90, 2, side="SELL") == 20


def test_csv_and_excel_save_helpers(tmp_path) -> None:
    rows = [{"symbol": "NIFTY", "pnl": 10.5}]

    csv_path = save_csv(rows, tmp_path / "nested" / "trades.csv")
    excel_path = save_excel(rows, tmp_path / "nested" / "trades.xlsx")

    assert csv_path.exists()
    assert excel_path.exists()


def test_logging_setup_creates_expected_files(tmp_path) -> None:
    setup_logging(log_dir=tmp_path, level="INFO", console=False)

    logging.getLogger("system-test").info("system")
    get_trade_logger().info("trade")
    get_broker_logger().info("broker")
    logging.getLogger("system-test").error("error")

    for filename in ("system.log", "error.log", "trade.log", "broker.log"):
        assert (tmp_path / filename).exists()

    with pytest.raises(ValueError):
        setup_logging(log_dir=tmp_path, level="NOPE", console=False)

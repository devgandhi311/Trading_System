"""Timezone-aware time helpers for Indian market hours."""

from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo

KOLKATA_TZ_NAME = "Asia/Kolkata"
KOLKATA_TZ = ZoneInfo(KOLKATA_TZ_NAME)
DEFAULT_MARKET_START = time(9, 15)
DEFAULT_MARKET_END = time(15, 30)
DEFAULT_NO_NEW_TRADE_AFTER = time(15, 0)
DEFAULT_FORCE_SQUAREOFF = time(15, 20)


def now_ist() -> datetime:
    """Return the current timezone-aware datetime in Asia/Kolkata."""

    return datetime.now(tz=KOLKATA_TZ)


def ensure_ist(value: datetime) -> datetime:
    """Return a datetime converted to Asia/Kolkata."""

    if value.tzinfo is None:
        return value.replace(tzinfo=KOLKATA_TZ)
    return value.astimezone(KOLKATA_TZ)


def parse_hhmm(value: str) -> time:
    """Parse a 24-hour HH:MM string into a time."""

    try:
        hour_text, minute_text = value.split(":", 1)
        parsed = time(int(hour_text), int(minute_text))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid HH:MM time: {value!r}") from exc
    return parsed


def combine_ist(day: date, value: time | str) -> datetime:
    """Combine a date and time as an Asia/Kolkata-aware datetime."""

    parsed_time = parse_hhmm(value) if isinstance(value, str) else value
    return datetime.combine(day, parsed_time, tzinfo=KOLKATA_TZ)


def is_weekday(value: datetime | date | None = None) -> bool:
    """Return true when the date is Monday through Friday."""

    candidate = ensure_ist(value).date() if isinstance(value, datetime) else value
    candidate = candidate or now_ist().date()
    return candidate.weekday() < 5


def is_market_open(
    value: datetime | None = None,
    *,
    market_start: time | str = DEFAULT_MARKET_START,
    market_end: time | str = DEFAULT_MARKET_END,
) -> bool:
    """Return true when the timestamp falls inside regular market hours."""

    candidate = ensure_ist(value or now_ist())
    if not is_weekday(candidate):
        return False
    start = parse_hhmm(market_start) if isinstance(market_start, str) else market_start
    end = parse_hhmm(market_end) if isinstance(market_end, str) else market_end
    return start <= candidate.time() <= end


def can_place_new_trade(
    value: datetime | None = None,
    *,
    market_start: time | str = DEFAULT_MARKET_START,
    no_new_trade_after: time | str = DEFAULT_NO_NEW_TRADE_AFTER,
) -> bool:
    """Return true when new trades are allowed at the timestamp."""

    candidate = ensure_ist(value or now_ist())
    if not is_weekday(candidate):
        return False
    start = parse_hhmm(market_start) if isinstance(market_start, str) else market_start
    cutoff = parse_hhmm(no_new_trade_after) if isinstance(no_new_trade_after, str) else no_new_trade_after
    return start <= candidate.time() <= cutoff


def should_force_squareoff(
    value: datetime | None = None,
    *,
    force_squareoff_time: time | str = DEFAULT_FORCE_SQUAREOFF,
) -> bool:
    """Return true once the force-squareoff time has been reached."""

    candidate = ensure_ist(value or now_ist())
    if not is_weekday(candidate):
        return False
    squareoff_time = parse_hhmm(force_squareoff_time) if isinstance(force_squareoff_time, str) else force_squareoff_time
    return candidate.time() >= squareoff_time


def seconds_until_market_open(
    value: datetime | None = None,
    *,
    market_start: time | str = DEFAULT_MARKET_START,
) -> float:
    """Return seconds until today's market open, or 0 if already open/past."""

    candidate = ensure_ist(value or now_ist())
    start_time = parse_hhmm(market_start) if isinstance(market_start, str) else market_start
    market_open = combine_ist(candidate.date(), start_time)
    return max(0.0, (market_open - candidate).total_seconds())

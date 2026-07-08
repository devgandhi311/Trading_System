"""Central logging setup for system, broker, trade, and error logs."""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from utils.file_utils import ensure_directory

LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_MAX_BYTES = 5_000_000
DEFAULT_BACKUP_COUNT = 5


def setup_logging(
    *,
    log_dir: str | Path = "logs",
    level: str | int = "INFO",
    console: bool = True,
) -> None:
    """Configure application logging with separate system, error, trade, and broker files."""

    directory = ensure_directory(log_dir)
    numeric_level = _resolve_level(level)
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.handlers.clear()
    root_logger.addHandler(
        _build_file_handler(directory / "system.log", numeric_level, formatter)
    )
    root_logger.addHandler(
        _build_file_handler(directory / "error.log", logging.ERROR, formatter)
    )
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    _configure_named_file_logger("trade", directory / "trade.log", numeric_level, formatter)
    _configure_named_file_logger("broker", directory / "broker.log", numeric_level, formatter)

    logging.getLogger(__name__).info("Logging initialized", extra={"log_dir": str(directory)})


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger by name."""

    return logging.getLogger(name)


def get_trade_logger() -> logging.Logger:
    """Return the dedicated trade logger."""

    return logging.getLogger("trade")


def get_broker_logger() -> logging.Logger:
    """Return the dedicated broker logger."""

    return logging.getLogger("broker")


def _build_file_handler(path: Path, level: int, formatter: logging.Formatter) -> RotatingFileHandler:
    handler = RotatingFileHandler(
        path,
        maxBytes=DEFAULT_MAX_BYTES,
        backupCount=DEFAULT_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


def _configure_named_file_logger(
    name: str,
    path: Path,
    level: int,
    formatter: logging.Formatter,
) -> None:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False
    logger.addHandler(_build_file_handler(path, level, formatter))


def _resolve_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    normalized = level.upper()
    if normalized not in logging._nameToLevel:
        raise ValueError(f"Unsupported log level: {level}")
    return logging._nameToLevel[normalized]

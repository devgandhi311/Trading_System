"""Typed configuration loading and validation for the trading system."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import re
from typing import Any, Mapping
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from models import Mode
from utils.exceptions import ConfigError
from utils.env_loader import load_project_env

LOGGER = logging.getLogger(__name__)


TradingMode = Mode


@dataclass(frozen=True)
class SettingsConfig:
    """Application-wide settings that are safe to keep in YAML."""

    mode: Mode
    live_trading_enabled: bool
    timezone: str
    market_start_time: str
    market_end_time: str
    no_new_trade_after_time: str
    force_squareoff_time: str
    log_level: str
    data_paths: dict[str, Path]
    report_paths: dict[str, Path]


@dataclass(frozen=True)
class RiskLimitsConfig:
    """Risk limits used by execution and risk checks."""

    capital: float
    max_daily_loss_percent: float
    max_daily_profit_lock_percent: float
    max_loss_per_trade_percent: float
    max_open_positions: int
    max_trades_per_day: int
    max_consecutive_losses: int
    max_margin_usage_percent: float
    max_bid_ask_spread_percent: float
    max_slippage_percent: float
    order_timeout_seconds: int
    emergency_market_exit_allowed: bool


@dataclass(frozen=True)
class StrategyConfig:
    """Strategy selection and sizing configuration."""

    allowed_strategies: tuple[str, ...]
    default_expiry_type: str
    min_confidence_score: float
    allowed_regimes: tuple[str, ...]
    lot_size: dict[str, int]
    max_lots_per_trade: int
    expected_move_credit_spread: dict[str, Any]


@dataclass(frozen=True)
class BrokerConfig:
    """Broker-level non-secret settings."""

    broker_name: str
    paper_broker: dict[str, Any]
    api: dict[str, Any]
    orders: dict[str, Any]


@dataclass(frozen=True)
class InstrumentConfig:
    """Tradable instrument metadata."""

    symbol: str
    exchange: str
    segment: str
    instrument_type: str
    enabled: bool
    lot_size: int
    tick_size: float


@dataclass(frozen=True)
class AppConfig:
    """Complete validated application configuration."""

    settings: SettingsConfig
    risk_limits: RiskLimitsConfig
    strategy: StrategyConfig
    broker: BrokerConfig
    instruments: tuple[InstrumentConfig, ...]


class ConfigLoader:
    """Load, validate, and expose YAML configuration plus .env secrets."""

    REQUIRED_FILES = {
        "settings": "settings.yaml",
        "risk_limits": "risk_limits.yaml",
        "strategy": "strategy_config.yaml",
        "broker": "broker_config.yaml",
        "instruments": "instruments.yaml",
    }
    SECRET_KEY_PATTERN = re.compile(
        r"(api[_-]?key|secret|token|password|passwd|pwd|access[_-]?key|client[_-]?secret)",
        re.IGNORECASE,
    )
    TIME_PATTERN = re.compile(r"^\d{2}:\d{2}$")
    VALID_LOG_LEVELS = {"CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"}

    def __init__(
        self,
        config_dir: str | Path = "config",
        env_file: str | Path = ".env",
        project_root: str | Path | None = None,
    ) -> None:
        """Create a loader rooted at the current project by default."""

        self.project_root = Path(project_root or Path.cwd()).resolve()
        self.config_dir = self._resolve_path(config_dir)
        self.env_file = self._resolve_path(env_file)
        load_project_env(self.env_file)
        self._raw_configs: dict[str, dict[str, Any]] = {}
        self._config: AppConfig | None = None
        self._secrets: dict[str, str] | None = None

    @property
    def config(self) -> AppConfig:
        """Return the loaded app config, loading it on first access."""

        return self.load()

    def load(self, reload: bool = False) -> AppConfig:
        """Load all YAML config files and return a validated AppConfig."""

        if self._config is not None and not reload:
            return self._config

        LOGGER.info("Loading configuration", extra={"config_dir": str(self.config_dir)})
        self._raw_configs = {
            name: self._read_yaml(self.config_dir / filename)
            for name, filename in self.REQUIRED_FILES.items()
        }
        self._reject_broker_secrets(self._raw_configs["broker"])

        settings = self._parse_settings(self._raw_configs["settings"])
        risk_limits = self._parse_risk_limits(self._raw_configs["risk_limits"])
        strategy = self._parse_strategy(self._raw_configs["strategy"])
        broker = self._parse_broker(self._raw_configs["broker"])
        instruments = self._parse_instruments(self._raw_configs["instruments"])
        self._validate_cross_config(settings, risk_limits, strategy, instruments)

        self._config = AppConfig(
            settings=settings,
            risk_limits=risk_limits,
            strategy=strategy,
            broker=broker,
            instruments=instruments,
        )
        LOGGER.info(
            "Configuration loaded",
            extra={"mode": settings.mode.value, "instruments": len(instruments)},
        )
        return self._config

    def raw(self, section: str) -> dict[str, Any]:
        """Return a copy of a raw loaded config section."""

        self.load()
        if section not in self._raw_configs:
            raise ConfigError(f"Unknown config section: {section}")
        return dict(self._raw_configs[section])

    def get_secret(self, key: str, *, required: bool = True) -> str | None:
        """Read a secret from .env only."""

        if not key:
            raise ConfigError("Secret key name must not be empty")
        secrets = self._load_env_file()
        value = secrets.get(key)
        if required and not value:
            raise ConfigError(f"Required secret {key!r} is missing from {self.env_file}")
        return value

    def _resolve_path(self, path: str | Path) -> Path:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = self.project_root / candidate
        return candidate.resolve()

    def _read_yaml(self, path: Path) -> dict[str, Any]:
        if not path.exists():
            raise ConfigError(f"Config file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as file:
                data = yaml.safe_load(file) or {}
        except yaml.YAMLError as exc:
            raise ConfigError(f"Invalid YAML in {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError(f"Config file must contain a mapping: {path}")
        return data

    def _load_env_file(self) -> dict[str, str]:
        if self._secrets is not None:
            return self._secrets
        secrets: dict[str, str] = {}
        if not self.env_file.exists():
            self._secrets = secrets
            return secrets
        for line_number, raw_line in enumerate(
            self.env_file.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ConfigError(f"Invalid .env entry at line {line_number}")
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if not key:
                raise ConfigError(f"Empty .env key at line {line_number}")
            secrets[key] = value
        self._secrets = secrets
        return secrets

    def _parse_settings(self, data: Mapping[str, Any]) -> SettingsConfig:
        mode = Mode(self._required_str(data, "mode").upper())
        live_trading_enabled = self._required_bool(data, "live_trading_enabled")
        timezone = self._required_str(data, "timezone")
        try:
            ZoneInfo(timezone)
        except ZoneInfoNotFoundError as exc:
            raise ConfigError(f"Unknown timezone: {timezone}") from exc
        if mode == Mode.LIVE and not live_trading_enabled:
            raise ConfigError("LIVE mode requires live_trading_enabled: true")
        if live_trading_enabled and mode != Mode.LIVE:
            raise ConfigError("live_trading_enabled can only be true in LIVE mode")

        market_start = self._required_time(data, "market_start_time")
        market_end = self._required_time(data, "market_end_time")
        no_new_trade_after = self._required_time(data, "no_new_trade_after_time")
        force_squareoff = self._required_time(data, "force_squareoff_time")
        self._ensure_time_order(
            (
                ("market_start_time", market_start),
                ("no_new_trade_after_time", no_new_trade_after),
                ("force_squareoff_time", force_squareoff),
                ("market_end_time", market_end),
            )
        )

        log_level = self._required_str(data, "log_level").upper()
        if log_level not in self.VALID_LOG_LEVELS:
            raise ConfigError(f"Unsupported log_level: {log_level}")

        return SettingsConfig(
            mode=mode,
            live_trading_enabled=live_trading_enabled,
            timezone=timezone,
            market_start_time=market_start,
            market_end_time=market_end,
            no_new_trade_after_time=no_new_trade_after,
            force_squareoff_time=force_squareoff,
            log_level=log_level,
            data_paths=self._required_path_map(data, "data_paths"),
            report_paths=self._required_path_map(data, "report_paths"),
        )

    def _parse_risk_limits(self, data: Mapping[str, Any]) -> RiskLimitsConfig:
        return RiskLimitsConfig(
            capital=self._required_positive_float(data, "capital"),
            max_daily_loss_percent=self._required_percent(data, "max_daily_loss_percent"),
            max_daily_profit_lock_percent=self._required_percent(data, "max_daily_profit_lock_percent"),
            max_loss_per_trade_percent=self._required_percent(data, "max_loss_per_trade_percent"),
            max_open_positions=self._required_positive_int(data, "max_open_positions"),
            max_trades_per_day=self._required_positive_int(data, "max_trades_per_day"),
            max_consecutive_losses=self._required_positive_int(data, "max_consecutive_losses"),
            max_margin_usage_percent=self._required_percent(data, "max_margin_usage_percent"),
            max_bid_ask_spread_percent=self._required_percent(data, "max_bid_ask_spread_percent"),
            max_slippage_percent=self._required_percent(data, "max_slippage_percent"),
            order_timeout_seconds=self._required_positive_int(data, "order_timeout_seconds"),
            emergency_market_exit_allowed=self._required_bool(data, "emergency_market_exit_allowed"),
        )

    def _parse_strategy(self, data: Mapping[str, Any]) -> StrategyConfig:
        return StrategyConfig(
            allowed_strategies=self._required_str_tuple(data, "allowed_strategies"),
            default_expiry_type=self._required_str(data, "default_expiry_type").upper(),
            min_confidence_score=self._required_float_between(data, "min_confidence_score", 0.0, 1.0),
            allowed_regimes=self._required_str_tuple(data, "allowed_regimes"),
            lot_size=self._required_int_map(data, "lot_size"),
            max_lots_per_trade=self._required_positive_int(data, "max_lots_per_trade"),
            expected_move_credit_spread=dict(data.get("expected_move_credit_spread", {})),
        )

    def _parse_broker(self, data: Mapping[str, Any]) -> BrokerConfig:
        return BrokerConfig(
            broker_name=self._required_str(data, "broker_name"),
            paper_broker=self._required_dict(data, "paper_broker"),
            api=self._required_dict(data, "api"),
            orders=self._required_dict(data, "orders"),
        )

    def _parse_instruments(self, data: Mapping[str, Any]) -> tuple[InstrumentConfig, ...]:
        raw_instruments = data.get("instruments")
        if not isinstance(raw_instruments, list) or not raw_instruments:
            raise ConfigError("instruments must be a non-empty list")
        instruments: list[InstrumentConfig] = []
        seen_symbols: set[str] = set()
        for index, item in enumerate(raw_instruments):
            if not isinstance(item, Mapping):
                raise ConfigError(f"Instrument at index {index} must be a mapping")
            symbol = self._required_str(item, "symbol").upper()
            if symbol in seen_symbols:
                raise ConfigError(f"Duplicate instrument symbol: {symbol}")
            seen_symbols.add(symbol)
            instruments.append(
                InstrumentConfig(
                    symbol=symbol,
                    exchange=self._required_str(item, "exchange"),
                    segment=self._required_str(item, "segment"),
                    instrument_type=self._required_str(item, "instrument_type"),
                    enabled=self._required_bool(item, "enabled"),
                    lot_size=self._required_positive_int(item, "lot_size"),
                    tick_size=self._required_positive_float(item, "tick_size"),
                )
            )
        return tuple(instruments)

    def _validate_cross_config(
        self,
        settings: SettingsConfig,
        risk_limits: RiskLimitsConfig,
        strategy: StrategyConfig,
        instruments: tuple[InstrumentConfig, ...],
    ) -> None:
        if risk_limits.max_open_positions > risk_limits.max_trades_per_day:
            raise ConfigError("max_open_positions cannot exceed max_trades_per_day")
        enabled_symbols = {instrument.symbol for instrument in instruments if instrument.enabled}
        missing_lot_sizes = enabled_symbols.difference(strategy.lot_size)
        if missing_lot_sizes:
            raise ConfigError(f"Missing strategy lot_size entries: {sorted(missing_lot_sizes)}")

    def _reject_broker_secrets(self, data: Mapping[str, Any], prefix: str = "broker_config") -> None:
        for key, value in data.items():
            current_path = f"{prefix}.{key}"
            if self.SECRET_KEY_PATTERN.search(str(key)):
                raise ConfigError(f"Secret-like broker setting must be moved to .env: {current_path}")
            if isinstance(value, Mapping):
                self._reject_broker_secrets(value, current_path)

    def _required_value(self, data: Mapping[str, Any], key: str) -> Any:
        if key not in data:
            raise ConfigError(f"Missing required config key: {key}")
        return data[key]

    def _required_str(self, data: Mapping[str, Any], key: str) -> str:
        value = self._required_value(data, key)
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{key} must be a non-empty string")
        return value.strip()

    def _required_bool(self, data: Mapping[str, Any], key: str) -> bool:
        value = self._required_value(data, key)
        if not isinstance(value, bool):
            raise ConfigError(f"{key} must be a boolean")
        return value

    def _required_positive_int(self, data: Mapping[str, Any], key: str) -> int:
        value = self._required_value(data, key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{key} must be a positive integer")
        return value

    def _required_positive_float(self, data: Mapping[str, Any], key: str) -> float:
        value = self._required_value(data, key)
        if not isinstance(value, int | float) or isinstance(value, bool) or value <= 0:
            raise ConfigError(f"{key} must be a positive number")
        return float(value)

    def _required_percent(self, data: Mapping[str, Any], key: str) -> float:
        return self._required_float_between(data, key, 0.0, 100.0)

    def _required_float_between(self, data: Mapping[str, Any], key: str, minimum: float, maximum: float) -> float:
        value = self._required_positive_float(data, key)
        if not minimum <= value <= maximum:
            raise ConfigError(f"{key} must be between {minimum} and {maximum}")
        return value

    def _required_time(self, data: Mapping[str, Any], key: str) -> str:
        value = self._required_str(data, key)
        if not self.TIME_PATTERN.match(value):
            raise ConfigError(f"{key} must use HH:MM format")
        hour, minute = (int(part) for part in value.split(":"))
        if hour > 23 or minute > 59:
            raise ConfigError(f"{key} must be a valid 24-hour time")
        return value

    def _required_dict(self, data: Mapping[str, Any], key: str) -> dict[str, Any]:
        value = self._required_value(data, key)
        if not isinstance(value, dict) or not value:
            raise ConfigError(f"{key} must be a non-empty mapping")
        return dict(value)

    def _required_path_map(self, data: Mapping[str, Any], key: str) -> dict[str, Path]:
        mapping = self._required_dict(data, key)
        resolved: dict[str, Path] = {}
        for item_key, item_value in mapping.items():
            if not isinstance(item_key, str) or not isinstance(item_value, str):
                raise ConfigError(f"{key} must map strings to path strings")
            resolved[item_key] = self._resolve_path(item_value)
        return resolved

    def _required_str_tuple(self, data: Mapping[str, Any], key: str) -> tuple[str, ...]:
        value = self._required_value(data, key)
        if not isinstance(value, list) or not value:
            raise ConfigError(f"{key} must be a non-empty list")
        parsed = tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
        if len(parsed) != len(value):
            raise ConfigError(f"{key} must contain only non-empty strings")
        return parsed

    def _required_int_map(self, data: Mapping[str, Any], key: str) -> dict[str, int]:
        mapping = self._required_dict(data, key)
        parsed: dict[str, int] = {}
        for item_key, item_value in mapping.items():
            if not isinstance(item_key, str):
                raise ConfigError(f"{key} keys must be strings")
            if not isinstance(item_value, int) or isinstance(item_value, bool) or item_value <= 0:
                raise ConfigError(f"{key}.{item_key} must be a positive integer")
            parsed[item_key.upper()] = item_value
        return parsed

    def _ensure_time_order(self, values: tuple[tuple[str, str], ...]) -> None:
        minutes = [(name, self._time_to_minutes(value)) for name, value in values]
        for (left_name, left_value), (right_name, right_value) in zip(minutes, minutes[1:]):
            if left_value >= right_value:
                raise ConfigError(f"{left_name} must be before {right_name}")

    @staticmethod
    def _time_to_minutes(value: str) -> int:
        hour, minute = (int(part) for part in value.split(":"))
        return hour * 60 + minute

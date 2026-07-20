"""Market data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass(frozen=True)
class DatasetManifest:
    """Immutable provenance and quality summary for a backtest dataset."""

    manifest_id: str
    data_source: str
    source_type: str
    source_file_paths: tuple[str, ...]
    source_file_hashes: dict[str, str]
    source_file_sizes: dict[str, int]
    detected_file_formats: dict[str, str]
    row_count: int
    first_timestamp: datetime
    last_timestamp: datetime
    detected_columns: tuple[str, ...]
    normalized_columns: tuple[str, ...]
    removed_rows: int
    duplicate_rows: int
    invalid_rows: int
    missing_value_counts: dict[str, int]
    timezone: str
    generated_sample: bool
    loader_version: str
    creation_timestamp: datetime

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable manifest representation."""

        return {
            "manifest_id": self.manifest_id,
            "data_source": self.data_source,
            "source_type": self.source_type,
            "source_file_paths": list(self.source_file_paths),
            "source_file_hashes": dict(self.source_file_hashes),
            "source_file_sizes": dict(self.source_file_sizes),
            "detected_file_formats": dict(self.detected_file_formats),
            "row_count": self.row_count,
            "first_timestamp": self.first_timestamp.isoformat(),
            "last_timestamp": self.last_timestamp.isoformat(),
            "detected_columns": list(self.detected_columns),
            "normalized_columns": list(self.normalized_columns),
            "removed_rows": self.removed_rows,
            "duplicate_rows": self.duplicate_rows,
            "invalid_rows": self.invalid_rows,
            "missing_value_counts": dict(self.missing_value_counts),
            "timezone": self.timezone,
            "generated_sample": self.generated_sample,
            "loader_version": self.loader_version,
            "creation_timestamp": self.creation_timestamp.isoformat(),
        }


@dataclass(frozen=True)
class MarketBar:
    """OHLCV market bar for a symbol and timeframe."""

    symbol: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int
    timeframe: str = "1m"
    open_interest: int | None = None
    vwap: float | None = None


@dataclass(frozen=True)
class OptionContract:
    """Option contract metadata and latest quote fields."""

    symbol: str
    underlying: str
    expiry: date
    strike: float
    option_type: str
    lot_size: int
    exchange: str = "NSE"
    bid: float | None = None
    ask: float | None = None
    last_price: float | None = None
    open_interest: int | None = None
    implied_volatility: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    instrument_token: str | None = None
    quote_timestamp: datetime | None = None
    available_quantity: int | None = None


@dataclass(frozen=True)
class OptionChainSnapshot:
    """Point-in-time option chain snapshot for an underlying."""

    underlying: str
    timestamp: datetime
    spot_price: float
    expiry: date
    contracts: tuple[OptionContract, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RequiredOptionLeg:
    """Exact option contract required by a candidate strategy."""

    expiry: date
    strike: float
    option_type: str


@dataclass(frozen=True)
class PaperMarketSnapshot:
    """Complete point-in-time input to strict PAPER market validation."""

    snapshot_id: str
    underlying: str
    timestamp: datetime
    spot_price: float
    spot_timestamp: datetime
    spot_instrument_token: str
    option_chain: OptionChainSnapshot
    futures_price: float | None = None
    futures_timestamp: datetime | None = None
    futures_instrument_token: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class MarketSnapshotValidationResult:
    """Fail-closed outcome of validating one PAPER market snapshot."""

    valid: bool
    entry_allowed: bool
    exit_allowed: bool
    blocking_reasons: tuple[str, ...]
    warnings: tuple[str, ...]
    snapshot_timestamp: datetime
    oldest_quote_age: float
    validated_instruments: tuple[str, ...]
    rejected_instruments: tuple[str, ...]

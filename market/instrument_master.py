"""Broker instrument master loading, saving, and lookup helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
import csv
import json
from pathlib import Path
from typing import Any, Protocol

from utils.exceptions import DataValidationError
from utils.file_utils import ensure_parent_dir
from utils.time_utils import now_ist


class InstrumentMasterAdapter(Protocol):
    """Protocol for broker instrument master download adapters."""

    def download_instruments(self) -> list[dict[str, Any]]:
        """Download instrument master rows from the broker."""


@dataclass(frozen=True)
class InstrumentRecord:
    """Normalized instrument master row."""

    symbol: str
    token: str
    underlying: str
    exchange: str
    segment: str
    instrument_type: str
    expiry: date | None = None
    strike: float | None = None
    option_type: str | None = None
    lot_size: int = 1
    tick_size: float = 0.05
    metadata: dict[str, Any] = field(default_factory=dict)


class InstrumentMaster:
    """Load, save, and query broker instrument master data."""

    def __init__(
        self,
        *,
        adapter: InstrumentMasterAdapter | None = None,
        storage_dir: str | Path = "data/raw/broker_downloads",
        filename: str = "instrument_master.csv",
    ) -> None:
        """Create an instrument master store."""

        self.adapter = adapter
        self.storage_path = Path(storage_dir) / filename
        self.records: tuple[InstrumentRecord, ...] = ()

    def download(self) -> tuple[InstrumentRecord, ...]:
        """Download instruments from adapter or create safe NIFTY/BANKNIFTY mocks."""

        if self.adapter is None:
            # TODO: Wire real Kotak instrument master download syntax here when
            # the approved SDK/API endpoint and schema are available.
            rows = self._mock_instruments()
        else:
            rows = self.adapter.download_instruments()
        self.records = tuple(self._record_from_mapping(row) for row in rows)
        self.save()
        return self.records

    def load(self, path: str | Path | None = None) -> tuple[InstrumentRecord, ...]:
        """Load instrument master from CSV or JSON."""

        file_path = Path(path or self.storage_path)
        if not file_path.exists():
            return self.download()
        if file_path.suffix.lower() == ".json":
            rows = json.loads(file_path.read_text(encoding="utf-8"))
        else:
            with file_path.open("r", encoding="utf-8", newline="") as file:
                rows = list(csv.DictReader(file))
        self.records = tuple(self._record_from_mapping(row) for row in rows)
        return self.records

    def save(self, path: str | Path | None = None) -> Path:
        """Save instrument master under data/raw/broker_downloads."""

        file_path = ensure_parent_dir(path or self.storage_path)
        with file_path.open("w", encoding="utf-8", newline="") as file:
            fieldnames = (
                "symbol",
                "token",
                "underlying",
                "exchange",
                "segment",
                "instrument_type",
                "expiry",
                "strike",
                "option_type",
                "lot_size",
                "tick_size",
            )
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(
                    {
                        "symbol": record.symbol,
                        "token": record.token,
                        "underlying": record.underlying,
                        "exchange": record.exchange,
                        "segment": record.segment,
                        "instrument_type": record.instrument_type,
                        "expiry": record.expiry.isoformat() if record.expiry else "",
                        "strike": "" if record.strike is None else record.strike,
                        "option_type": record.option_type or "",
                        "lot_size": record.lot_size,
                        "tick_size": record.tick_size,
                    }
                )
        return file_path

    def lookup_by_symbol(self, symbol: str) -> InstrumentRecord | None:
        """Lookup an instrument by symbol."""

        normalized = symbol.upper()
        return next((record for record in self.records if record.symbol.upper() == normalized), None)

    def lookup_by_token(self, token: str) -> InstrumentRecord | None:
        """Lookup an instrument by broker token."""

        normalized = str(token)
        return next((record for record in self.records if str(record.token) == normalized), None)

    def lookup_options(
        self,
        *,
        underlying: str,
        expiry: date | None = None,
        strike: float | None = None,
        option_type: str | None = None,
    ) -> tuple[InstrumentRecord, ...]:
        """Lookup option instruments by underlying, expiry, strike, and CE/PE."""

        normalized_underlying = underlying.upper()
        normalized_type = option_type.upper() if option_type else None
        return tuple(
            record
            for record in self.records
            if record.underlying.upper() == normalized_underlying
            and record.instrument_type.upper() in {"OPTIDX", "OPTION"}
            and (expiry is None or record.expiry == expiry)
            and (strike is None or record.strike == float(strike))
            and (normalized_type is None or record.option_type == normalized_type)
        )

    def expiries(self, underlying: str) -> tuple[date, ...]:
        """Return sorted option expiries for an underlying."""

        values = {
            record.expiry
            for record in self.lookup_options(underlying=underlying)
            if record.expiry is not None
        }
        return tuple(sorted(values))

    def _record_from_mapping(self, row: dict[str, Any]) -> InstrumentRecord:
        symbol = str(row.get("symbol") or row.get("tradingsymbol") or "").upper()
        token = str(row.get("token") or row.get("instrument_token") or symbol)
        underlying = str(row.get("underlying") or row.get("name") or symbol).upper()
        if not symbol:
            raise DataValidationError("Instrument row missing symbol")
        expiry = self._parse_date(row.get("expiry"))
        option_type = row.get("option_type") or row.get("optionType")
        return InstrumentRecord(
            symbol=symbol,
            token=token,
            underlying=underlying,
            exchange=str(row.get("exchange") or "NSE"),
            segment=str(row.get("segment") or "NFO"),
            instrument_type=str(row.get("instrument_type") or row.get("instrumentType") or "OPTIDX").upper(),
            expiry=expiry,
            strike=self._optional_float(row.get("strike")),
            option_type=str(option_type).upper() if option_type else None,
            lot_size=int(float(row.get("lot_size") or row.get("lotSize") or 1)),
            tick_size=float(row.get("tick_size") or row.get("tickSize") or 0.05),
            metadata=dict(row),
        )

    @staticmethod
    def _parse_date(value: Any) -> date | None:
        if value in (None, ""):
            return None
        if isinstance(value, date):
            return value
        return date.fromisoformat(str(value))

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        return float(value)

    @staticmethod
    def _mock_instruments() -> list[dict[str, Any]]:
        today = now_ist().date()
        first_expiry = today + timedelta(days=(3 - today.weekday()) % 7 or 7)
        second_expiry = first_expiry + timedelta(days=7)
        rows: list[dict[str, Any]] = [
            {"symbol": "NIFTY", "token": "NIFTY", "underlying": "NIFTY", "segment": "INDEX", "instrument_type": "INDEX", "lot_size": 75},
            {"symbol": "BANKNIFTY", "token": "BANKNIFTY", "underlying": "BANKNIFTY", "segment": "INDEX", "instrument_type": "INDEX", "lot_size": 35},
        ]
        for underlying, step, center, lot_size in (("NIFTY", 50, 25000, 75), ("BANKNIFTY", 100, 56000, 35)):
            for expiry in (first_expiry, second_expiry):
                for strike in range(center - step * 3, center + step * 4, step):
                    for option_type in ("CE", "PE"):
                        rows.append(
                            {
                                "symbol": f"{underlying}{expiry:%y%m%d}{strike}{option_type}",
                                "token": f"{underlying}-{expiry:%Y%m%d}-{strike}-{option_type}",
                                "underlying": underlying,
                                "exchange": "NFO",
                                "segment": "NFO-OPT",
                                "instrument_type": "OPTIDX",
                                "expiry": expiry.isoformat(),
                                "strike": strike,
                                "option_type": option_type,
                                "lot_size": lot_size,
                                "tick_size": 0.05,
                            }
                        )
        return rows

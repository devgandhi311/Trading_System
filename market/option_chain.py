"""Option-chain construction and strike selection helpers."""

from __future__ import annotations

from dataclasses import asdict
from datetime import date
import json
from pathlib import Path
from typing import Any

from market.instrument_master import InstrumentMaster, InstrumentRecord
from models.market_models import OptionChainSnapshot, OptionContract
from utils.exceptions import DataValidationError
from utils.file_utils import ensure_parent_dir, save_csv
from utils.math_utils import round_to_tick
from utils.time_utils import now_ist


class OptionChain:
    """Build option-chain snapshots for NIFTY and BANKNIFTY."""

    SUPPORTED_UNDERLYINGS = {"NIFTY", "BANKNIFTY"}

    def __init__(self, instrument_master: InstrumentMaster | None = None) -> None:
        """Create an option-chain helper."""

        self.instrument_master = instrument_master or InstrumentMaster()
        if not self.instrument_master.records:
            self.instrument_master.load()

    def get_nearest_expiry(self, underlying: str, as_of: date | None = None) -> date:
        """Return the nearest available expiry on or after as_of."""

        self._validate_underlying(underlying)
        current = as_of or now_ist().date()
        expiries = [expiry for expiry in self.instrument_master.expiries(underlying) if expiry >= current]
        if not expiries:
            raise DataValidationError(f"No expiries available for {underlying}")
        return min(expiries)

    def get_monthly_expiry(self, underlying: str, as_of: date | None = None) -> date:
        """Return the last available expiry in the nearest expiry month."""

        self._validate_underlying(underlying)
        current = as_of or now_ist().date()
        expiries = [expiry for expiry in self.instrument_master.expiries(underlying) if expiry >= current]
        if not expiries:
            raise DataValidationError(f"No monthly expiry available for {underlying}")
        first_month = min(expiries).replace(day=1)
        monthly = [expiry for expiry in expiries if expiry.year == first_month.year and expiry.month == first_month.month]
        return max(monthly)

    def get_custom_expiry(self, underlying: str, expiry: date | str) -> date:
        """Validate and return a custom expiry."""

        self._validate_underlying(underlying)
        parsed = date.fromisoformat(expiry) if isinstance(expiry, str) else expiry
        if parsed not in self.instrument_master.expiries(underlying):
            raise DataValidationError(f"Expiry {parsed} is not available for {underlying}")
        return parsed

    def get_atm_strike(self, spot_price: float, strike_step: int | float) -> float:
        """Return nearest ATM strike for a spot price."""

        return round_to_tick(float(spot_price), float(strike_step))

    def select_otm_strike(
        self,
        *,
        spot_price: float,
        option_type: str,
        steps: int = 1,
        strike_step: int | float,
    ) -> float:
        """Select an OTM strike for CE or PE."""

        atm = self.get_atm_strike(spot_price, strike_step)
        offset = steps * float(strike_step)
        return atm + offset if option_type.upper() == "CE" else atm - offset

    def select_itm_strike(
        self,
        *,
        spot_price: float,
        option_type: str,
        steps: int = 1,
        strike_step: int | float,
    ) -> float:
        """Select an ITM strike for CE or PE."""

        atm = self.get_atm_strike(spot_price, strike_step)
        offset = steps * float(strike_step)
        return atm - offset if option_type.upper() == "CE" else atm + offset

    def build_option_chain_snapshot(
        self,
        *,
        underlying: str,
        spot_price: float,
        expiry: date | str | None = None,
        quotes: dict[str, dict[str, Any]] | None = None,
    ) -> OptionChainSnapshot:
        """Build an option-chain snapshot from instrument master and optional quotes."""

        self._validate_underlying(underlying)
        parsed_expiry = self.get_custom_expiry(underlying, expiry) if expiry else self.get_nearest_expiry(underlying)
        records = self.instrument_master.lookup_options(underlying=underlying, expiry=parsed_expiry)
        contracts = tuple(self._contract_from_record(record, quotes or {}) for record in records)
        return OptionChainSnapshot(
            underlying=underlying.upper(),
            timestamp=now_ist(),
            spot_price=float(spot_price),
            expiry=parsed_expiry,
            contracts=contracts,
            metadata={"source": "instrument_master"},
        )

    def save_option_chain_snapshot(self, snapshot: OptionChainSnapshot, path: str | Path) -> Path:
        """Save option-chain snapshot as JSON or CSV."""

        file_path = ensure_parent_dir(path)
        rows = [
            {
                **asdict(contract),
                "expiry": contract.expiry.isoformat(),
                "snapshot_underlying": snapshot.underlying,
                "snapshot_time": snapshot.timestamp.isoformat(),
                "spot_price": snapshot.spot_price,
            }
            for contract in snapshot.contracts
        ]
        if file_path.suffix.lower() == ".json":
            payload = {
                "underlying": snapshot.underlying,
                "timestamp": snapshot.timestamp.isoformat(),
                "spot_price": snapshot.spot_price,
                "expiry": snapshot.expiry.isoformat(),
                "contracts": rows,
                "metadata": snapshot.metadata,
            }
            file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            return file_path
        return save_csv(rows, file_path)

    def _contract_from_record(self, record: InstrumentRecord, quotes: dict[str, dict[str, Any]]) -> OptionContract:
        quote = quotes.get(record.symbol, quotes.get(record.token, {}))
        return OptionContract(
            symbol=record.symbol,
            underlying=record.underlying,
            expiry=record.expiry or now_ist().date(),
            strike=float(record.strike or 0.0),
            option_type=str(record.option_type or ""),
            lot_size=record.lot_size,
            exchange=record.exchange,
            bid=self._optional_float(quote.get("bid")),
            ask=self._optional_float(quote.get("ask")),
            last_price=self._optional_float(quote.get("last_price") or quote.get("ltp")),
            open_interest=self._optional_int(quote.get("open_interest") or quote.get("oi")),
            implied_volatility=self._optional_float(quote.get("implied_volatility") or quote.get("iv")),
            metadata={"token": record.token},
        )

    def _validate_underlying(self, underlying: str) -> None:
        if underlying.upper() not in self.SUPPORTED_UNDERLYINGS:
            raise DataValidationError(f"Unsupported underlying: {underlying}")

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return None if value in (None, "") else float(value)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if value in (None, "") else int(value)

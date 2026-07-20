"""Authoritative PAPER market input adapter; synthetic data is opt-in only."""
from __future__ import annotations
from datetime import datetime
import hashlib
from typing import Callable
from market.market_data_manager import MarketDataManager
from models.market_models import PaperMarketSnapshot
from runtime.paper_trading_engine import PaperCycleInputs
from utils.exceptions import DataValidationError
from utils.time_utils import ensure_ist, now_ist

class PaperMarketInputProvider:
    def __init__(self, manager: MarketDataManager, *, underlying: str = "NIFTY", allow_demo: bool = False,
                 clock: Callable[[], datetime] = now_ist):
        self.manager, self.underlying, self.allow_demo, self.clock = manager, underlying, allow_demo, clock
    def __call__(self) -> PaperCycleInputs:
        now = ensure_ist(self.clock())
        tick = self.manager.get_latest_tick(self.underlying)
        chain = self.manager.get_latest_option_chain(self.underlying)
        if tick is None: raise DataValidationError("Authoritative PAPER spot tick unavailable")
        source = str(tick.get("source") or chain.metadata.get("source") or "").upper()
        if not self.allow_demo and source in {"", "MOCK", "SAMPLE", "SYNTHETIC", "DEMO"}:
            raise DataValidationError("Synthetic or unproven PAPER market source is prohibited")
        raw_timestamp = tick.get("timestamp") or now
        if isinstance(raw_timestamp, str):
            try:
                raw_timestamp = datetime.fromisoformat(raw_timestamp.replace("Z", "+00:00"))
            except ValueError as exc:
                raise DataValidationError("Invalid authoritative tick timestamp") from exc
        timestamp = ensure_ist(raw_timestamp)
        token = str(tick.get("instrument_token") or tick.get("token") or "")
        spot = float(tick.get("last_price") or 0.0)
        identity = f"{self.underlying}|{timestamp.isoformat()}|{spot}|{chain.expiry.isoformat()}"
        snapshot = PaperMarketSnapshot(hashlib.sha256(identity.encode()).hexdigest(), self.underlying, timestamp,
            spot, timestamp, token, chain, metadata={"source": source, "as_of": timestamp.isoformat()})
        validation = self.manager.validate_paper_snapshot(snapshot, current_time=now)
        if not validation.valid: raise DataValidationError("; ".join(validation.blocking_reasons))
        bars = self.manager.historical_data if self.manager.historical_data is not None else self.manager.load_historical_data()
        if bars is None or bars.empty: raise DataValidationError("Authoritative historical feature input unavailable")
        return PaperCycleInputs(snapshot, bars)

"""Unified access to historical data, live ticks, option chains, and broker quotes."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

import pandas as pd

from broker.kotak_client import KotakClient
from broker.websocket_feed import WebSocketFeed
from market.data_loader import DataLoader
from market.data_store import DataStore
from market.option_chain import OptionChain
from models import Mode
from models.market_models import OptionChainSnapshot
from utils.config_loader import ConfigLoader
from utils.exceptions import DataValidationError
from utils.file_utils import ensure_parent_dir
from utils.time_utils import now_ist

if TYPE_CHECKING:
    from features.feature_pipeline import FeaturePipeline, FeaturePipelineResult


class MarketDataManager:
    """Facade for historical, live, option-chain, and quote data."""

    def __init__(
        self,
        *,
        mode: Mode | str | None = None,
        config_loader: ConfigLoader | None = None,
        data_loader: DataLoader | None = None,
        data_store: DataStore | None = None,
        websocket_feed: WebSocketFeed | None = None,
        broker_client: KotakClient | None = None,
        option_chain: OptionChain | None = None,
        feature_pipeline: "FeaturePipeline | None" = None,
    ) -> None:
        """Create a market data manager using configured defaults."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        self.mode = Mode(str(mode).upper()) if mode is not None else self.config.settings.mode
        self.data_store = data_store or DataStore(config_loader=self.config_loader)
        self.data_loader = data_loader or DataLoader(config_loader=self.config_loader, data_store=self.data_store)
        self.websocket_feed = websocket_feed or WebSocketFeed(mode=self.mode, config_loader=self.config_loader)
        self.broker_client = broker_client or KotakClient(mode=self.mode, config_loader=self.config_loader)
        self.option_chain = option_chain or OptionChain()
        if feature_pipeline is None:
            from features.feature_pipeline import FeaturePipeline

            feature_pipeline = FeaturePipeline(mode=self.mode, data_store=self.data_store)
        self.feature_pipeline = feature_pipeline
        self.historical_data: pd.DataFrame | None = None
        self.latest_option_chain: OptionChainSnapshot | None = None

    def load_historical_data(self, dataset_name: str = "nse_30_years_clean") -> pd.DataFrame:
        """Load historical data from store or raw files."""

        try:
            self.historical_data = self.data_store.load_processed(dataset_name)
        except DataValidationError:
            self.historical_data, _ = self.data_loader.load_30_years_nse_data(dataset_name=dataset_name)
        return self.historical_data

    def get_latest_spot(self, symbol: str = "NIFTY") -> float:
        """Return latest spot price from historical, tick, or broker data by mode."""

        if self.mode == Mode.BACKTEST:
            data = self.historical_data if self.historical_data is not None else self.load_historical_data()
            if data.empty:
                raise DataValidationError("Historical data is empty")
            return float(data.sort_values("date").iloc[-1]["close"])
        tick = self.get_latest_tick(symbol)
        if tick is not None:
            return float(tick["last_price"])
        return float(self.broker_client.get_ltp(symbol))

    def get_latest_future(self, symbol: str = "NIFTY") -> float:
        """Return latest future price, using quote fallback when no dedicated future feed exists."""

        future_symbol = f"{symbol.upper()}_FUT"
        tick = self.get_latest_tick(future_symbol)
        if tick is not None:
            return float(tick["last_price"])
        if self.mode == Mode.BACKTEST:
            return self.get_latest_spot(symbol)
        try:
            return float(self.broker_client.get_ltp(future_symbol))
        except Exception:
            return self.get_latest_spot(symbol)

    def get_latest_option_chain(
        self,
        underlying: str = "NIFTY",
        *,
        spot_price: float | None = None,
    ) -> OptionChainSnapshot:
        """Return the current option-chain snapshot, refreshing if needed."""

        if self.latest_option_chain is None or self.latest_option_chain.underlying != underlying.upper():
            return self.refresh_option_chain(underlying=underlying, spot_price=spot_price)
        return self.latest_option_chain

    def get_latest_tick(self, symbol_or_token: str) -> dict[str, Any] | None:
        """Return latest tick by symbol/token from the live feed cache."""

        return self.websocket_feed.latest_ticks.get(symbol_or_token.upper())

    def refresh_option_chain(
        self,
        underlying: str = "NIFTY",
        *,
        spot_price: float | None = None,
    ) -> OptionChainSnapshot:
        """Refresh and cache an option-chain snapshot."""

        spot = float(spot_price if spot_price is not None else self.get_latest_spot(underlying))
        self.latest_option_chain = self.option_chain.build_option_chain_snapshot(
            underlying=underlying,
            spot_price=spot,
        )
        return self.latest_option_chain

    def save_live_snapshot(self, path: str | Path = "data/live/market_snapshot.json") -> Path:
        """Save latest ticks and option-chain state to disk."""

        file_path = ensure_parent_dir(path)
        payload = {
            "timestamp": now_ist().isoformat(),
            "mode": self.mode.value,
            "latest_ticks": self.websocket_feed.latest_ticks,
            "option_chain": self._serialize_option_chain(self.latest_option_chain),
        }
        file_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return file_path

    def generate_features(
        self,
        dataframe: pd.DataFrame | None = None,
        *,
        dataset_name: str = "combined_features",
    ) -> "FeaturePipelineResult":
        """Generate cached features using the unified feature pipeline."""

        source = dataframe
        if source is None:
            source = self.historical_data if self.historical_data is not None else self.load_historical_data()
        return self.feature_pipeline.generate_features(source, dataset_name=dataset_name)

    @staticmethod
    def _serialize_option_chain(snapshot: OptionChainSnapshot | None) -> dict[str, Any] | None:
        if snapshot is None:
            return None
        data = asdict(snapshot) if is_dataclass(snapshot) else dict(snapshot)
        data["timestamp"] = snapshot.timestamp.isoformat()
        data["expiry"] = snapshot.expiry.isoformat()
        for contract in data["contracts"]:
            contract["expiry"] = contract["expiry"].isoformat() if hasattr(contract["expiry"], "isoformat") else contract["expiry"]
        return data

"""Market data fetching, broker instruments, and option-chain helpers."""

from market.data_loader import DataLoader, DataQualityReport
from market.data_preprocessor import DataPreprocessor
from market.data_store import DataStore
from market.instrument_master import InstrumentMaster, InstrumentMasterAdapter, InstrumentRecord
from market.option_chain import OptionChain

__all__ = [
    "DataLoader",
    "MarketDataManager",
    "DataPreprocessor",
    "DataQualityReport",
    "DataStore",
    "InstrumentMaster",
    "InstrumentMasterAdapter",
    "InstrumentRecord",
    "OptionChain",
]


def __getattr__(name: str):
    """Lazily expose heavier integrations that depend on features and broker modules."""

    if name == "MarketDataManager":
        from market.market_data_manager import MarketDataManager

        return MarketDataManager
    raise AttributeError(name)

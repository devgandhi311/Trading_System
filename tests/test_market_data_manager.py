import json

import pandas as pd

from broker.websocket_feed import WebSocketFeed
from market.data_store import DataStore
from market.market_data_manager import MarketDataManager
from models import Mode


def test_backtest_manager_uses_historical_data(tmp_path) -> None:
    store = DataStore(
        processed_dir=tmp_path / "processed",
        features_dir=tmp_path / "features",
        models_dir=tmp_path / "models",
    )
    store.save_processed(
        pd.DataFrame(
            [
                {"date": "2024-01-01", "open": 1, "high": 2, "low": 0.5, "close": 1.5, "volume": 10},
                {"date": "2024-01-02", "open": 2, "high": 3, "low": 1.5, "close": 2.5, "volume": 10},
            ]
        ),
        "nse_30_years_clean",
        prefer_parquet=False,
    )
    manager = MarketDataManager(mode=Mode.BACKTEST, data_store=store)

    assert manager.get_latest_spot("NIFTY") == 2.5
    assert manager.get_latest_future("NIFTY") == 2.5


def test_paper_manager_uses_latest_tick_and_saves_snapshot(tmp_path) -> None:
    feed = WebSocketFeed(mode=Mode.PAPER, latest_ticks_path=tmp_path / "latest_ticks.json")
    feed.generate_mock_tick("NIFTY", price=25123.0)
    manager = MarketDataManager(mode=Mode.PAPER, websocket_feed=feed)

    assert manager.get_latest_tick("NIFTY")["last_price"] == 25123.0
    assert manager.get_latest_spot("NIFTY") == 25123.0
    snapshot_path = manager.save_live_snapshot(tmp_path / "snapshot.json")

    payload = json.loads(snapshot_path.read_text(encoding="utf-8"))
    assert payload["latest_ticks"]["NIFTY"]["last_price"] == 25123.0

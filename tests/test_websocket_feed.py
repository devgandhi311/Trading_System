import json

import pytest

from broker.websocket_feed import WebSocketFeed
from models import Mode
from utils.exceptions import BrokerError


def test_paper_feed_subscribe_persists_tick_and_calls_callback(tmp_path) -> None:
    ticks = []
    feed = WebSocketFeed(
        mode=Mode.PAPER,
        callback=ticks.append,
        latest_ticks_path=tmp_path / "latest_ticks.json",
    )

    feed.subscribe(["NIFTY"])

    assert feed.connected
    assert feed.heartbeat.last_heartbeat_at is not None
    assert ticks
    saved = json.loads((tmp_path / "latest_ticks.json").read_text(encoding="utf-8"))
    assert saved["NIFTY"]["last_price"] == 25000.0


def test_reconnect_triggers_kill_switch_after_threshold(tmp_path) -> None:
    feed = WebSocketFeed(
        mode=Mode.PAPER,
        latest_ticks_path=tmp_path / "latest_ticks.json",
    )
    feed.max_disconnects = 1
    feed.subscribe(["NIFTY"])
    feed.reconnect()

    with pytest.raises(BrokerError, match="kill switch"):
        feed.reconnect()

    assert feed.heartbeat.kill_switch_triggered


def test_unsubscribe_clears_mock_subscriptions(tmp_path) -> None:
    feed = WebSocketFeed(mode=Mode.BACKTEST, latest_ticks_path=tmp_path / "latest_ticks.json")
    feed.subscribe(["NIFTY", "BANKNIFTY"])

    feed.unsubscribe()

    assert not feed.connected
    assert feed.subscriptions == set()

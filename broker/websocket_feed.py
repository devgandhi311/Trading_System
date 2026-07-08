"""WebSocket market data feed with safe mock ticks for non-live modes."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Any, Protocol

from models import Mode
from utils.config_loader import ConfigLoader
from utils.exceptions import BrokerError
from utils.file_utils import ensure_parent_dir
from utils.time_utils import now_ist

LOGGER = logging.getLogger("broker")
TickCallback = Callable[[dict[str, Any]], None]


class WebSocketAdapter(Protocol):
    """Protocol for a real Kotak WebSocket adapter."""

    def connect(self) -> None:
        """Open the underlying WebSocket connection."""

    def subscribe(self, symbols_or_tokens: tuple[str, ...]) -> None:
        """Subscribe to symbols or tokens."""

    def unsubscribe(self, symbols_or_tokens: tuple[str, ...] | None = None) -> None:
        """Unsubscribe from symbols or tokens, or all when None."""

    def close(self) -> None:
        """Close the WebSocket connection."""


@dataclass(frozen=True)
class FeedHeartbeat:
    """WebSocket heartbeat and disconnect state."""

    connected: bool
    last_heartbeat_at: datetime | None
    disconnect_count: int
    kill_switch_triggered: bool


class WebSocketFeed:
    """Manage broker WebSocket subscriptions and latest tick persistence."""

    def __init__(
        self,
        *,
        mode: Mode | str | None = None,
        adapter: WebSocketAdapter | None = None,
        config_loader: ConfigLoader | None = None,
        callback: TickCallback | None = None,
        latest_ticks_path: str | Path | None = None,
    ) -> None:
        """Create a feed; PAPER/BACKTEST use safe mock ticks."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        self.mode = Mode(str(mode).upper()) if mode is not None else self.config.settings.mode
        self.adapter = adapter
        self.callback = callback
        websocket_config = self.config.broker.api.get("websocket") or self.config.broker.paper_broker.get("websocket") or {}
        raw_broker = self.config_loader.raw("broker")
        websocket_config = raw_broker.get("websocket", websocket_config)
        self.max_disconnects = int(websocket_config.get("max_disconnects_before_kill_switch", 5))
        self.latest_ticks_path = Path(
            latest_ticks_path or websocket_config.get("latest_ticks_path", "data/live/latest_ticks.json")
        )
        self.subscriptions: set[str] = set()
        self.latest_ticks: dict[str, dict[str, Any]] = {}
        self.connected = False
        self.disconnect_count = 0
        self.kill_switch_triggered = False
        self.last_heartbeat_at: datetime | None = None

    @property
    def heartbeat(self) -> FeedHeartbeat:
        """Return current heartbeat state."""

        return FeedHeartbeat(
            connected=self.connected,
            last_heartbeat_at=self.last_heartbeat_at,
            disconnect_count=self.disconnect_count,
            kill_switch_triggered=self.kill_switch_triggered,
        )

    @property
    def is_mock_mode(self) -> bool:
        """Return true for PAPER/BACKTEST mock feeds."""

        return self.mode in {Mode.PAPER, Mode.BACKTEST}

    def subscribe(self, symbols_or_tokens: Iterable[str], callback: TickCallback | None = None) -> None:
        """Subscribe to one or more symbols/tokens."""

        symbols = tuple(str(symbol).upper() for symbol in symbols_or_tokens if str(symbol).strip())
        if not symbols:
            raise BrokerError("At least one symbol or token is required for subscription")
        if callback is not None:
            self.callback = callback
        if self.kill_switch_triggered:
            raise BrokerError("WebSocket kill switch is active")
        self.subscriptions.update(symbols)
        LOGGER.info("WebSocket subscribe requested", extra={"count": len(symbols), "mode": self.mode.value})
        if self.is_mock_mode:
            self.connected = True
            self.update_heartbeat()
            for symbol in symbols:
                self.generate_mock_tick(symbol)
            return
        if self.adapter is None:
            # TODO: Wire actual Kotak WebSocket subscribe syntax here when the
            # approved SDK/API method signatures are available.
            raise BrokerError("Kotak WebSocket adapter is not configured")
        self.adapter.connect()
        self.adapter.subscribe(symbols)
        self.connected = True
        self.update_heartbeat()

    def unsubscribe(self, symbols_or_tokens: Iterable[str] | None = None) -> None:
        """Unsubscribe from selected symbols/tokens or all subscriptions."""

        symbols = None
        if symbols_or_tokens is not None:
            symbols = tuple(str(symbol).upper() for symbol in symbols_or_tokens if str(symbol).strip())
            self.subscriptions.difference_update(symbols)
        else:
            self.subscriptions.clear()
        LOGGER.info("WebSocket unsubscribe requested", extra={"mode": self.mode.value})
        if self.is_mock_mode:
            self.connected = False
            return
        if self.adapter is None:
            raise BrokerError("Kotak WebSocket adapter is not configured")
        self.adapter.unsubscribe(symbols)
        if not self.subscriptions:
            self.connected = False

    def reconnect(self) -> None:
        """Reconnect the feed and enforce disconnect kill-switch limits."""

        self.disconnect_count += 1
        LOGGER.warning("WebSocket reconnect requested", extra={"disconnect_count": self.disconnect_count})
        if self.disconnect_count > self.max_disconnects:
            self.kill_switch_triggered = True
            self.connected = False
            raise BrokerError("WebSocket disconnect kill switch triggered")
        subscriptions = tuple(self.subscriptions)
        self.connected = False
        if self.adapter is not None and not self.is_mock_mode:
            self.adapter.close()
        if subscriptions:
            self.subscribe(subscriptions)

    def update_heartbeat(self) -> None:
        """Update the last heartbeat timestamp."""

        self.last_heartbeat_at = now_ist()
        LOGGER.debug("WebSocket heartbeat updated", extra={"mode": self.mode.value})

    def on_tick(self, tick: dict[str, Any]) -> None:
        """Process, persist, and callback a single tick."""

        normalized = self._normalize_tick(tick)
        key = str(normalized["symbol_or_token"]).upper()
        self.latest_ticks[key] = normalized
        self.update_heartbeat()
        self._persist_latest_ticks()
        if self.callback is not None:
            self.callback(normalized)

    def generate_mock_tick(self, symbol_or_token: str, price: float | None = None) -> dict[str, Any]:
        """Generate and process a deterministic mock tick."""

        base_prices = {"NIFTY": 25000.0, "BANKNIFTY": 56000.0}
        symbol = str(symbol_or_token).upper()
        tick = {
            "symbol_or_token": symbol,
            "timestamp": now_ist().isoformat(),
            "last_price": float(price if price is not None else base_prices.get(symbol, 100.0)),
            "bid": float(price if price is not None else base_prices.get(symbol, 100.0)) - 0.05,
            "ask": float(price if price is not None else base_prices.get(symbol, 100.0)) + 0.05,
            "volume": 0,
            "mode": self.mode.value,
            "mock": True,
        }
        self.on_tick(tick)
        return tick

    def _normalize_tick(self, tick: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(tick, dict):
            raise BrokerError("Tick must be a mapping")
        symbol = tick.get("symbol_or_token") or tick.get("symbol") or tick.get("token")
        if not symbol:
            raise BrokerError("Tick missing symbol_or_token")
        last_price = tick.get("last_price", tick.get("ltp"))
        if last_price is None:
            raise BrokerError("Tick missing last_price")
        normalized = dict(tick)
        normalized["symbol_or_token"] = str(symbol).upper()
        normalized["last_price"] = float(last_price)
        normalized["timestamp"] = str(tick.get("timestamp") or now_ist().isoformat())
        return normalized

    def _persist_latest_ticks(self) -> None:
        path = ensure_parent_dir(self.latest_ticks_path)
        suffix = path.suffix.lower()
        rows = list(self.latest_ticks.values())
        if suffix == ".json":
            path.write_text(json.dumps(self.latest_ticks, indent=2, sort_keys=True), encoding="utf-8")
            return
        if suffix == ".csv":
            import csv

            fieldnames = sorted({key for row in rows for key in row})
            with path.open("w", encoding="utf-8", newline="") as file:
                writer = csv.DictWriter(file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            return
        if suffix == ".parquet":
            try:
                import pandas as pd
            except ImportError as exc:
                raise BrokerError("pandas is required to persist latest ticks as parquet") from exc
            pd.DataFrame(rows).to_parquet(path, index=False)
            return
        raise BrokerError(f"Unsupported latest tick storage format: {suffix}")

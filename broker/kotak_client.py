"""Read-only Kotak broker client with safe paper/backtest mocks."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import logging
from typing import Any, Protocol

from broker.kotak_auth import KotakAuth
from models import Mode, OrderSide, OrderStatus
from models.broker_models import (
    BrokerProfile,
    HoldingSnapshot,
    MarginSnapshot,
    OrderSnapshot,
    PositionSnapshot,
    QuoteSnapshot,
)
from utils.config_loader import ConfigLoader
from utils.exceptions import BrokerError
from utils.time_utils import now_ist

LOGGER = logging.getLogger("broker")


class KotakDataAdapter(Protocol):
    """Protocol for real Kotak account and market-data SDK/API calls."""

    def get_profile(self) -> Any:
        """Return broker profile data."""

    def get_margin(self) -> Any:
        """Return margin data."""

    def get_positions(self) -> Any:
        """Return positions data."""

    def get_orders(self) -> Any:
        """Return orders data."""

    def get_order_status(self, order_id: str) -> Any:
        """Return a single order status."""

    def get_holdings(self) -> Any:
        """Return holdings data."""

    def get_ltp(self, symbol_or_token: str) -> Any:
        """Return last traded price."""

    def get_quote(self, symbol_or_token: str) -> Any:
        """Return full quote data."""


class KotakClient:
    """Read-only Kotak client for account and market-data functions."""

    def __init__(
        self,
        *,
        auth: KotakAuth | None = None,
        data_adapter: KotakDataAdapter | None = None,
        mode: Mode | str | None = None,
        config_loader: ConfigLoader | None = None,
        mock_prices: dict[str, float] | None = None,
    ) -> None:
        """Create a read-only Kotak client.

        PAPER and BACKTEST modes use deterministic mocks. LIVE mode requires
        a real adapter for account and market-data functions.
        """

        self.config_loader = config_loader
        self.auth = auth or KotakAuth()
        self.data_adapter = data_adapter
        self.mode = self._resolve_mode(mode)
        self.mock_prices = {
            "NIFTY": 25000.0,
            "BANKNIFTY": 56000.0,
            "FINNIFTY": 24000.0,
            "MIDCPNIFTY": 13000.0,
            **(mock_prices or {}),
        }
        LOGGER.info("Kotak client initialized", extra={"mode": self.mode.value})

    @property
    def is_mock_mode(self) -> bool:
        """Return true when the client must avoid real broker data calls."""

        return self.mode in {Mode.PAPER, Mode.BACKTEST}

    def login(self) -> bool:
        """Login for live mode or mark mock modes as ready."""

        if self.is_mock_mode:
            LOGGER.info("Kotak mock login completed", extra={"mode": self.mode.value})
            return True
        self.auth.login()
        LOGGER.info("Kotak live login completed")
        return True

    def validate_session(self) -> bool:
        """Validate live session or mock readiness."""

        if self.is_mock_mode:
            LOGGER.info("Kotak mock session validated", extra={"mode": self.mode.value})
            return True
        return self.auth.validate_session()

    def get_profile(self) -> BrokerProfile:
        """Return account profile data."""

        if self.is_mock_mode:
            return BrokerProfile(
                user_id=f"{self.mode.value.lower()}-user",
                broker_name="kotak-mock",
                account_type=self.mode.value,
                display_name="Mock Kotak User",
            )
        response = self._call_live("get_profile")
        data = self._as_mapping(response)
        return BrokerProfile(
            user_id=str(data.get("user_id") or data.get("client_id") or "unknown"),
            broker_name="kotak",
            account_type=str(data.get("account_type") or "LIVE"),
            display_name=self._optional_str(data.get("display_name") or data.get("name")),
            metadata=data,
        )

    def get_margin(self) -> MarginSnapshot:
        """Return margin snapshot."""

        if self.is_mock_mode:
            balance = self._mock_initial_balance()
            return MarginSnapshot(
                timestamp=now_ist(),
                cash=balance,
                available_margin=balance,
                used_margin=0.0,
                opening_balance=balance,
            )
        data = self._as_mapping(self._call_live("get_margin"))
        return MarginSnapshot(
            timestamp=now_ist(),
            cash=self._float_value(data, "cash", "net_cash", default=0.0),
            available_margin=self._float_value(data, "available_margin", "available", default=0.0),
            used_margin=self._float_value(data, "used_margin", "utilized", default=0.0),
            opening_balance=self._float_value(data, "opening_balance", "balance", default=0.0),
            metadata=data,
        )

    def get_positions(self) -> tuple[PositionSnapshot, ...]:
        """Return current positions."""

        if self.is_mock_mode:
            return ()
        rows = self._as_sequence(self._call_live("get_positions"))
        return tuple(self._position_from_mapping(self._as_mapping(row)) for row in rows)

    def get_orders(self) -> tuple[OrderSnapshot, ...]:
        """Return read-only orders."""

        if self.is_mock_mode:
            return ()
        rows = self._as_sequence(self._call_live("get_orders"))
        return tuple(self._order_from_mapping(self._as_mapping(row)) for row in rows)

    def get_order_status(self, order_id: str) -> OrderSnapshot:
        """Return read-only status for an order."""

        if not order_id:
            raise BrokerError("order_id is required")
        if self.is_mock_mode:
            return OrderSnapshot(
                order_id=order_id,
                symbol="MOCK",
                side=OrderSide.BUY,
                quantity=0,
                status=OrderStatus.PENDING,
                timestamp=now_ist(),
            )
        return self._order_from_mapping(self._as_mapping(self._call_live("get_order_status", order_id)))

    def get_holdings(self) -> tuple[HoldingSnapshot, ...]:
        """Return account holdings."""

        if self.is_mock_mode:
            return ()
        rows = self._as_sequence(self._call_live("get_holdings"))
        return tuple(self._holding_from_mapping(self._as_mapping(row)) for row in rows)

    def get_ltp(self, symbol_or_token: str) -> float:
        """Return last traded price for a symbol or token."""

        if not symbol_or_token:
            raise BrokerError("symbol_or_token is required")
        if self.is_mock_mode:
            return self._mock_price(symbol_or_token)
        response = self._call_live("get_ltp", symbol_or_token)
        if isinstance(response, int | float):
            return float(response)
        data = self._as_mapping(response)
        return self._float_value(data, "last_price", "ltp", "price", default=0.0)

    def get_quote(self, symbol_or_token: str) -> QuoteSnapshot:
        """Return quote snapshot for a symbol or token."""

        if not symbol_or_token:
            raise BrokerError("symbol_or_token is required")
        if self.is_mock_mode:
            price = self._mock_price(symbol_or_token)
            return QuoteSnapshot(
                symbol_or_token=symbol_or_token,
                timestamp=now_ist(),
                last_price=price,
                bid=price - 0.05,
                ask=price + 0.05,
                open=price,
                high=price,
                low=price,
                close=price,
                volume=0,
                metadata={"mode": self.mode.value, "mock": True},
            )
        data = self._as_mapping(self._call_live("get_quote", symbol_or_token))
        return QuoteSnapshot(
            symbol_or_token=symbol_or_token,
            timestamp=now_ist(),
            last_price=self._float_value(data, "last_price", "ltp", "price", default=0.0),
            bid=self._optional_float(data.get("bid")),
            ask=self._optional_float(data.get("ask")),
            open=self._optional_float(data.get("open")),
            high=self._optional_float(data.get("high")),
            low=self._optional_float(data.get("low")),
            close=self._optional_float(data.get("close")),
            volume=self._optional_int(data.get("volume")),
            metadata=data,
        )

    def _resolve_mode(self, mode: Mode | str | None) -> Mode:
        if mode is not None:
            return Mode(str(mode).upper())
        if self.config_loader is not None:
            return self.config_loader.load().settings.mode
        return ConfigLoader().load().settings.mode

    def _call_live(self, method_name: str, *args: Any) -> Any:
        if self.data_adapter is None:
            # TODO: Wire real Kotak SDK/API account and market-data calls through
            # a KotakDataAdapter implementation when approved syntax is available.
            LOGGER.error("Kotak data adapter unavailable", extra={"method": method_name})
            raise BrokerError(f"Kotak data adapter is not configured for {method_name}")
        if not self.auth.validate_session():
            raise BrokerError("Kotak session is invalid")
        try:
            method = getattr(self.data_adapter, method_name)
            response = method(*args)
        except BrokerError:
            raise
        except Exception as exc:
            LOGGER.error("Kotak data call failed", extra={"method": method_name})
            raise BrokerError(f"Kotak data call failed: {method_name}") from exc
        LOGGER.info("Kotak data call succeeded", extra={"method": method_name})
        return response

    def _mock_initial_balance(self) -> float:
        if self.config_loader is not None:
            config = self.config_loader.load()
        else:
            config = ConfigLoader().load()
        return float(config.broker.paper_broker.get("initial_balance", config.risk_limits.capital))

    def _mock_price(self, symbol_or_token: str) -> float:
        return float(self.mock_prices.get(symbol_or_token.upper(), 100.0))

    def _position_from_mapping(self, data: dict[str, Any]) -> PositionSnapshot:
        quantity = self._int_value(data, "quantity", "qty", default=0)
        average_price = self._float_value(data, "average_price", "avg_price", default=0.0)
        last_price = self._float_value(data, "last_price", "ltp", default=average_price)
        return PositionSnapshot(
            symbol=str(data.get("symbol") or data.get("tradingsymbol") or data.get("token") or "unknown"),
            quantity=quantity,
            average_price=average_price,
            last_price=last_price,
            unrealized_pnl=self._float_value(data, "unrealized_pnl", "pnl", default=0.0),
            product_type=str(data.get("product_type") or data.get("product") or "MIS"),
            metadata=data,
        )

    def _order_from_mapping(self, data: dict[str, Any]) -> OrderSnapshot:
        return OrderSnapshot(
            order_id=str(data.get("order_id") or data.get("id") or ""),
            symbol=str(data.get("symbol") or data.get("tradingsymbol") or "unknown"),
            side=OrderSide(str(data.get("side") or data.get("transaction_type") or "BUY").upper()),
            quantity=self._int_value(data, "quantity", "qty", default=0),
            status=OrderStatus(str(data.get("status") or "PENDING").upper()),
            timestamp=now_ist(),
            average_price=self._optional_float(data.get("average_price") or data.get("avg_price")),
            filled_quantity=self._int_value(data, "filled_quantity", "filled_qty", default=0),
            metadata=data,
        )

    def _holding_from_mapping(self, data: dict[str, Any]) -> HoldingSnapshot:
        average_price = self._float_value(data, "average_price", "avg_price", default=0.0)
        last_price = self._float_value(data, "last_price", "ltp", default=average_price)
        quantity = self._int_value(data, "quantity", "qty", default=0)
        return HoldingSnapshot(
            symbol=str(data.get("symbol") or data.get("tradingsymbol") or "unknown"),
            quantity=quantity,
            average_price=average_price,
            last_price=last_price,
            unrealized_pnl=(last_price - average_price) * quantity,
            metadata=data,
        )

    @staticmethod
    def _as_mapping(value: Any) -> dict[str, Any]:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, dict):
            return dict(value)
        if hasattr(value, "__dict__"):
            return dict(vars(value))
        raise BrokerError(f"Kotak response must be mapping-like, got {type(value).__name__}")

    @staticmethod
    def _as_sequence(value: Any) -> tuple[Any, ...]:
        if value is None:
            return ()
        if isinstance(value, tuple):
            return value
        if isinstance(value, list):
            return tuple(value)
        raise BrokerError(f"Kotak response must be a sequence, got {type(value).__name__}")

    @staticmethod
    def _float_value(data: dict[str, Any], *keys: str, default: float) -> float:
        for key in keys:
            value = data.get(key)
            if value is not None:
                return float(value)
        return default

    @staticmethod
    def _int_value(data: dict[str, Any], *keys: str, default: int) -> int:
        for key in keys:
            value = data.get(key)
            if value is not None:
                return int(value)
        return default

    @staticmethod
    def _optional_float(value: Any) -> float | None:
        return None if value is None else float(value)

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if value is None else int(value)

    @staticmethod
    def _optional_str(value: Any) -> str | None:
        return None if value is None else str(value)

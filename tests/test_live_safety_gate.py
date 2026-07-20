"""Live gate integration and runtime-mode isolation tests."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from broker.order_manager import OrderManager
from execution.order_router import OrderBasket, OrderRouter
from models import Mode, OrderSide, OrderType
from models.order_models import OrderRequest
from utils.exceptions import ExecutionError


class ConfigLoader:
    def __init__(self, mode: Mode, enabled: bool = True):
        self.mode = mode
        self.enabled = enabled

    def load(self):
        return SimpleNamespace(
            settings=SimpleNamespace(
                mode=self.mode,
                live_trading_enabled=self.enabled if self.mode == Mode.LIVE else False,
                live_order_submission_enabled=self.enabled if self.mode == Mode.LIVE else False,
            ),
            risk_limits=SimpleNamespace(
                emergency_market_exit_allowed=True,
                max_bid_ask_spread_percent=1.0,
                max_slippage_percent=0.2,
                order_timeout_seconds=30,
            ),
            broker=SimpleNamespace(orders={"product_type": "MIS", "validity": "DAY"}),
        )


class Adapter:
    def __init__(self):
        self.calls = 0

    def place_order(self, request):
        self.calls += 1
        return {"order_id": "live-1", "status": "SUBMITTED", "symbol": request.symbol,
                "side": request.side, "quantity": request.quantity}


def request(key="gate-1"):
    return OrderRequest(
        symbol="NIFTY26000CE", side=OrderSide.BUY, quantity=75,
        order_type=OrderType.LIMIT, timestamp=datetime.now(timezone.utc), limit_price=100.0,
        metadata={"idempotency_key": key},
    )


def quote(_symbol):
    return {"ltp": 100.0, "bid": 99.9, "ask": 100.1}


def test_live_safety_rejection_prevents_adapter_invocation() -> None:
    loader, adapter = ConfigLoader(Mode.LIVE), Adapter()
    manager = OrderManager(config_loader=loader, execution_adapter=adapter)
    router = OrderRouter(
        config_loader=loader,
        order_manager=manager,
        live_safety_approval=lambda: (_ for _ in ()).throw(ExecutionError("gate rejected")),
    )
    with pytest.raises(ExecutionError, match="gate rejected"):
        router.place_order(request(), quote)
    assert adapter.calls == 0


def test_live_safety_approval_allows_routing() -> None:
    loader, adapter = ConfigLoader(Mode.LIVE), Adapter()
    manager = OrderManager(config_loader=loader, execution_adapter=adapter)
    router = OrderRouter(config_loader=loader, order_manager=manager, live_safety_approval=lambda: object())
    assert router.place_order(request(), quote).order_id == "live-1"
    assert adapter.calls == 1


@pytest.mark.parametrize("mode", [Mode.PAPER, Mode.BACKTEST])
def test_simulated_modes_never_call_live_adapter(mode) -> None:
    loader, adapter = ConfigLoader(mode), Adapter()
    manager = OrderManager(config_loader=loader, mode=mode, execution_adapter=adapter)
    response = manager.place_order(request(str(mode)))
    assert response.raw_response["simulated"] is True
    assert adapter.calls == 0


def test_dependent_basket_legs_are_blocked_after_unknown() -> None:
    class TimeoutAdapter(Adapter):
        def place_order(self, request):
            self.calls += 1
            raise TimeoutError("unknown")

    loader, adapter = ConfigLoader(Mode.LIVE), TimeoutAdapter()
    manager = OrderManager(config_loader=loader, execution_adapter=adapter)
    router = OrderRouter(config_loader=loader, order_manager=manager, live_safety_approval=lambda: object())
    basket = OrderBasket("strategy-1", (request("leg-1"), request("leg-2")))
    with pytest.raises(Exception, match="UNKNOWN"):
        router.place_basket(basket, quote)
    assert adapter.calls == 1


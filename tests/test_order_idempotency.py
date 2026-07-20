"""Deterministic order idempotency and unknown-state tests."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from types import SimpleNamespace
import logging
import time

import pytest

from broker.order_manager import OrderManager
from execution.retry_handler import RetryHandler, RetryPolicy, RetryablePreSubmissionError
from models import Mode, OrderSide, OrderType
from models.order_models import OrderIntentStatus, OrderRequest
from utils.exceptions import BrokerError


class LiveConfigLoader:
    def load(self):
        return SimpleNamespace(
            settings=SimpleNamespace(
                mode=Mode.LIVE,
                live_trading_enabled=True,
                live_order_submission_enabled=True,
            ),
            risk_limits=SimpleNamespace(emergency_market_exit_allowed=True),
        )


def order(key: str = "intent-1") -> OrderRequest:
    return OrderRequest(
        symbol="NIFTY26000CE",
        side=OrderSide.BUY,
        quantity=75,
        order_type=OrderType.LIMIT,
        timestamp=datetime.now(timezone.utc),
        limit_price=100.0,
        metadata={"idempotency_key": key},
    )


class Adapter:
    def __init__(self, *, fail=False, reconciliation=None, delay=0.0):
        self.calls = 0
        self.fail = fail
        self.reconciliation = reconciliation
        self.delay = delay

    def place_order(self, request):
        self.calls += 1
        if self.delay:
            time.sleep(self.delay)
        if self.fail:
            raise TimeoutError("ambiguous timeout token=never-log-this")
        return {
            "order_id": "broker-1",
            "status": "SUBMITTED",
            "symbol": request.symbol,
            "side": request.side,
            "quantity": request.quantity,
        }

    def find_order_by_idempotency_key(self, key):
        if isinstance(self.reconciliation, Exception):
            raise self.reconciliation
        return self.reconciliation


def manager(adapter: Adapter) -> OrderManager:
    return OrderManager(config_loader=LiveConfigLoader(), execution_adapter=adapter)


def test_duplicate_intent_calls_broker_once() -> None:
    adapter = Adapter()
    subject = manager(adapter)
    first = subject.place_order(order())
    second = subject.place_order(order())
    assert first is second
    assert adapter.calls == 1


def test_concurrent_duplicate_calls_broker_once() -> None:
    adapter = Adapter(delay=0.03)
    subject = manager(adapter)
    with ThreadPoolExecutor(max_workers=8) as pool:
        responses = list(pool.map(lambda _: subject.place_order(order()), range(8)))
    assert adapter.calls == 1
    assert {response.order_id for response in responses} == {"broker-1"}


def test_timeout_after_invocation_becomes_unknown_and_cannot_retry() -> None:
    adapter = Adapter(fail=True)
    subject = manager(adapter)
    with pytest.raises(BrokerError, match="UNKNOWN"):
        subject.place_order(order())
    assert subject.order_intents["intent-1"].status == OrderIntentStatus.UNKNOWN
    with pytest.raises(BrokerError, match="automatic retry blocked"):
        subject.place_order(order())
    assert adapter.calls == 1


def test_confirmed_not_found_permits_one_controlled_retry() -> None:
    adapter = Adapter(fail=True, reconciliation=None)
    subject = manager(adapter)
    with pytest.raises(BrokerError):
        subject.place_order(order())
    assert subject.reconcile_order_intent("intent-1").status == OrderIntentStatus.RETRY_ALLOWED
    adapter.fail = False
    assert subject.place_order(order()).order_id == "broker-1"
    assert adapter.calls == 2
    assert subject.place_order(order()).order_id == "broker-1"
    assert adapter.calls == 2


def test_ambiguous_reconciliation_remains_blocked() -> None:
    adapter = Adapter(fail=True, reconciliation=TimeoutError("still unknown"))
    subject = manager(adapter)
    with pytest.raises(BrokerError):
        subject.place_order(order())
    record = subject.reconcile_order_intent("intent-1")
    assert record.status == OrderIntentStatus.RECONCILIATION_REQUIRED
    with pytest.raises(BrokerError, match="reconciliation"):
        subject.place_order(order())
    assert adapter.calls == 1


def test_logs_redact_secrets(caplog) -> None:
    adapter = Adapter()
    subject = manager(adapter)
    request = order("secret-log")
    request.metadata.update({"api_key": "TOP-SECRET", "authorization": "Bearer SECRET"})
    with caplog.at_level(logging.INFO):
        subject.place_order(request)
    rendered = repr([(record.getMessage(), record.__dict__) for record in caplog.records])
    assert "TOP-SECRET" not in rendered
    assert "Bearer SECRET" not in rendered
    assert "***" in rendered


def test_only_proven_pre_submission_failure_is_retried() -> None:
    handler = RetryHandler(RetryPolicy(attempts=2, backoff_seconds=0))
    pre_calls = 0

    def pre_submission():
        nonlocal pre_calls
        pre_calls += 1
        if pre_calls == 1:
            raise RetryablePreSubmissionError("socket unavailable before invocation")
        return "ok"

    assert handler.run(pre_submission, operation_name="preflight") == "ok"
    assert pre_calls == 2

    ambiguous_calls = 0

    def ambiguous():
        nonlocal ambiguous_calls
        ambiguous_calls += 1
        raise TimeoutError("adapter may have accepted order")

    with pytest.raises(TimeoutError):
        handler.run(ambiguous, operation_name="place_order")
    assert ambiguous_calls == 1


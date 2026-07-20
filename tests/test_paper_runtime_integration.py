from __future__ import annotations

from types import SimpleNamespace

from runtime.paper_trading_engine import PaperTradingEngine


def test_strategy_instance_key_is_stable_across_decision_wall_clock():
    value = lambda name: SimpleNamespace(value=name)
    context = SimpleNamespace(decision_cycle_id="first-cycle")
    leg = SimpleNamespace(symbol="OPT", side=value("BUY"), quantity=50,
                          order_type=value("LIMIT"), limit_price=100, trigger_price=None)
    plan = SimpleNamespace(strategy_type=value("LONG_OPTION"), symbol="NIFTY", legs=[leg])
    snapshot = SimpleNamespace(snapshot_id="immutable-snapshot")
    first = PaperTradingEngine._strategy_instance_key("s1", context, plan, snapshot)
    context.decision_cycle_id = "a-restarted-cycle-id"
    second = PaperTradingEngine._strategy_instance_key("s1", context, plan, snapshot)
    assert first == second

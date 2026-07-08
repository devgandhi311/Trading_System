from datetime import date, datetime, timezone

from models import (
    LatencyReport,
    MarginSnapshot,
    MarketBar,
    Mode,
    OptionChainSnapshot,
    OptionContract,
    OrderRequest,
    OrderResponse,
    OrderSide,
    OrderStatus,
    OrderType,
    PredictionResult,
    QuoteSnapshot,
    RegimeResult,
    RegimeType,
    RiskDecision,
    StrategyLeg,
    StrategyPlan,
    StrategyType,
    TradeReport,
    TradeState,
)
from utils.config_loader import TradingMode


def test_shared_enums_are_available() -> None:
    assert Mode.PAPER.value == "PAPER"
    assert TradingMode is Mode
    assert OrderSide.BUY.value == "BUY"
    assert OrderType.SL_LIMIT.value == "SL_LIMIT"
    assert OrderStatus.FILLED.value == "FILLED"
    assert StrategyType.MOMENTUM_BREAKOUT.value == "momentum_breakout"
    assert RegimeType.RANGE_BOUND.value == "range_bound"


def test_shared_dataclasses_can_be_instantiated() -> None:
    now = datetime.now(timezone.utc)
    expiry = date(2026, 7, 30)
    contract = OptionContract(
        symbol="NIFTY26JUL25000CE",
        underlying="NIFTY",
        expiry=expiry,
        strike=25000.0,
        option_type="CE",
        lot_size=75,
    )
    leg = StrategyLeg(symbol=contract.symbol, side=OrderSide.BUY, quantity=75)

    instances = [
        MarketBar("NIFTY", now, 1.0, 2.0, 0.5, 1.5, 1000),
        contract,
        OptionChainSnapshot("NIFTY", now, 25000.0, expiry, (contract,)),
        RegimeResult("NIFTY", now, RegimeType.TRENDING, 0.8, 50),
        PredictionResult("NIFTY", now, "UP", 0.75, 15, RegimeType.TRENDING),
        leg,
        StrategyPlan("s1", StrategyType.MOMENTUM_BREAKOUT, "NIFTY", now, (leg,), 0.75, "test"),
        RiskDecision(True, now, "approved"),
        OrderRequest(contract.symbol, OrderSide.BUY, 75, OrderType.MARKET, now),
        OrderResponse("o1", OrderStatus.FILLED, now, contract.symbol, OrderSide.BUY, 75),
        TradeState("t1", contract.symbol, OrderSide.BUY, 75, now, 100.0, OrderStatus.OPEN),
        TradeReport(expiry, now, 1, 1, 0, 100.0, 95.0),
        LatencyReport(expiry, now, "execution", 10, 1.0, 2.0, 3.0, 4.0),
        MarginSnapshot(now, 100000.0, 90000.0, 10000.0, 100000.0),
        QuoteSnapshot("NIFTY", now, 25000.0),
    ]

    assert len(instances) == 15

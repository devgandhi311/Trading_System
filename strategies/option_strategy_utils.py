"""Shared option strategy helpers for advisory plan construction."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from models import OrderSide, OrderType, StrategyLeg
from models.market_models import OptionChainSnapshot, OptionContract


@dataclass(frozen=True)
class StrategyMetrics:
    """Calculated strategy metrics stored in StrategyPlan metadata."""

    max_profit: float
    max_loss: float
    breakeven: tuple[float, ...]
    target_profit: float
    stop_loss: float
    margin_estimate_placeholder: float

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly metric payload."""

        return {
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "breakeven": self.breakeven,
            "target_profit": self.target_profit,
            "stop_loss": self.stop_loss,
            "margin_estimate_placeholder": self.margin_estimate_placeholder,
        }


def contracts_by_type(snapshot: OptionChainSnapshot, option_type: str) -> list[OptionContract]:
    """Return sorted contracts for CE or PE."""

    normalized = option_type.upper()
    return sorted(
        [contract for contract in snapshot.contracts if contract.option_type.upper() == normalized],
        key=lambda contract: contract.strike,
    )


def nearest_otm(snapshot: OptionChainSnapshot, option_type: str, steps: int = 1) -> OptionContract | None:
    """Return an OTM option near spot, with steps moving farther OTM."""

    contracts = contracts_by_type(snapshot, option_type)
    if not contracts:
        return None
    if option_type.upper() == "CE":
        candidates = [contract for contract in contracts if contract.strike > snapshot.spot_price]
    else:
        candidates = [contract for contract in reversed(contracts) if contract.strike < snapshot.spot_price]
    if not candidates:
        return min(contracts, key=lambda contract: abs(contract.strike - snapshot.spot_price))
    index = min(max(steps - 1, 0), len(candidates) - 1)
    return candidates[index]


def farther_otm(snapshot: OptionChainSnapshot, short_contract: OptionContract, steps: int = 1) -> OptionContract | None:
    """Return a farther OTM hedge contract from a short option."""

    contracts = contracts_by_type(snapshot, short_contract.option_type)
    if short_contract.option_type.upper() == "CE":
        candidates = [contract for contract in contracts if contract.strike > short_contract.strike]
    else:
        candidates = [contract for contract in reversed(contracts) if contract.strike < short_contract.strike]
    if not candidates:
        return None
    index = min(max(steps - 1, 0), len(candidates) - 1)
    return candidates[index]


def atm_contract(snapshot: OptionChainSnapshot, option_type: str) -> OptionContract | None:
    """Return nearest ATM contract for CE or PE."""

    contracts = contracts_by_type(snapshot, option_type)
    if not contracts:
        return None
    return min(contracts, key=lambda contract: abs(contract.strike - snapshot.spot_price))


def option_price(contract: OptionContract, side: OrderSide) -> float:
    """Return executable advisory price for a side."""

    if side == OrderSide.BUY:
        return float(contract.ask or contract.last_price or contract.bid or 0.0)
    return float(contract.bid or contract.last_price or contract.ask or 0.0)


def make_leg(contract: OptionContract, side: OrderSide, quantity: int) -> StrategyLeg:
    """Create a limit StrategyLeg from an option contract."""

    price = option_price(contract, side)
    return StrategyLeg(
        symbol=contract.symbol,
        side=side,
        quantity=quantity,
        order_type=OrderType.LIMIT,
        limit_price=price,
        metadata={
            "underlying": contract.underlying,
            "expiry": contract.expiry.isoformat(),
            "strike": contract.strike,
            "option_type": contract.option_type.upper(),
        },
    )


def net_premium(legs: Iterable[StrategyLeg]) -> float:
    """Return net premium points, positive for credit and negative for debit."""

    total = 0.0
    quantity = None
    for leg in legs:
        quantity = quantity or max(leg.quantity, 1)
        price = float(leg.limit_price or 0.0)
        total += price if leg.side == OrderSide.SELL else -price
    return total


def total_cash_premium(legs: Iterable[StrategyLeg]) -> float:
    """Return absolute premium cash exposure."""

    return sum(float(leg.limit_price or 0.0) * leg.quantity for leg in legs)


def strike_width(first: OptionContract, second: OptionContract) -> float:
    """Return absolute strike width."""

    return abs(float(first.strike) - float(second.strike))


def missing_chain_plan_reason(snapshot: OptionChainSnapshot | None) -> str | None:
    """Return a validation reason when option-chain input is unusable."""

    if snapshot is None:
        return "Option chain snapshot is required."
    if not snapshot.contracts:
        return "Option chain snapshot has no contracts."
    return None

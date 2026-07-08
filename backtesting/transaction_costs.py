"""Transaction cost, slippage, and fill simulation for backtests."""

from __future__ import annotations

from dataclasses import dataclass

from models import OrderSide


@dataclass(frozen=True)
class TransactionCostModel:
    """Approximate brokerage, taxes, spread, and slippage costs."""

    brokerage_per_order: float = 20.0
    exchange_fee_percent: float = 0.003
    tax_percent: float = 0.02
    slippage_percent: float = 0.05

    def estimate(self, *, price: float, quantity: int, orders: int = 1) -> float:
        """Estimate total cost for traded notional."""

        notional = abs(price * quantity)
        variable_cost = notional * (self.exchange_fee_percent + self.tax_percent + self.slippage_percent) / 100.0
        return variable_cost + self.brokerage_per_order * max(1, orders)

    def apply_slippage(self, *, price: float, side: OrderSide, is_entry: bool = True) -> float:
        """Apply directional slippage to a simulated fill price."""

        adjustment = price * self.slippage_percent / 100.0
        if side == OrderSide.BUY:
            return price + adjustment if is_entry else price - adjustment
        return price - adjustment if is_entry else price + adjustment


@dataclass(frozen=True)
class FillResult:
    """Simulated order fill result."""

    filled: bool
    fill_price: float | None
    reason: str


class FillSimulator:
    """Simulate limit and SL-LIMIT fills from OHLC bars."""

    def limit_fill(self, *, side: OrderSide, limit_price: float, high: float, low: float) -> FillResult:
        """Return whether a limit order would fill inside a bar."""

        if side == OrderSide.BUY and low <= limit_price:
            return FillResult(True, limit_price, "buy_limit_touched")
        if side == OrderSide.SELL and high >= limit_price:
            return FillResult(True, limit_price, "sell_limit_touched")
        return FillResult(False, None, "limit_not_touched")

    def sl_limit_fill(
        self,
        *,
        side: OrderSide,
        trigger_price: float,
        limit_price: float,
        high: float,
        low: float,
    ) -> FillResult:
        """Return whether an SL-LIMIT order would trigger and fill."""

        triggered = high >= trigger_price if side == OrderSide.BUY else low <= trigger_price
        if not triggered:
            return FillResult(False, None, "trigger_not_touched")
        return self.limit_fill(side=side, limit_price=limit_price, high=high, low=low)


def approximate_option_payoff(
    *,
    spot_price: float,
    strike: float,
    premium: float,
    option_type: str,
    side: OrderSide,
    quantity: int,
) -> float:
    """Approximate expiry payoff for a vanilla option leg."""

    option = option_type.upper()
    intrinsic = max(0.0, spot_price - strike) if option == "CE" else max(0.0, strike - spot_price)
    buyer_pnl = (intrinsic - premium) * quantity
    return buyer_pnl if side == OrderSide.BUY else -buyer_pnl

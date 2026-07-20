"""Strict, deterministic PAPER market snapshot validation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Mapping

from models.market_models import (
    MarketSnapshotValidationResult,
    OptionContract,
    PaperMarketSnapshot,
    RequiredOptionLeg,
)
from utils.time_utils import now_ist


@dataclass(frozen=True)
class MarketSnapshotValidationPolicy:
    """Safety thresholds for snapshot freshness and consistency."""

    max_quote_age_seconds: float = 5.0
    max_option_chain_age_seconds: float = 10.0
    max_underlying_option_skew_seconds: float = 3.0
    max_price_jump_percent: float = 20.0
    require_futures: bool = False


class MarketSnapshotValidator:
    """Validate PAPER inputs without substituting mock or fallback data."""

    def __init__(self, policy: MarketSnapshotValidationPolicy | None = None) -> None:
        self.policy = policy or MarketSnapshotValidationPolicy()

    def validate(
        self,
        snapshot: PaperMarketSnapshot,
        *,
        required_legs: tuple[RequiredOptionLeg, ...] = (),
        previous_prices: Mapping[str, float] | None = None,
        current_time: datetime | None = None,
        emergency_exit: bool = False,
    ) -> MarketSnapshotValidationResult:
        """Return a deterministic fail-closed validation result."""

        now = current_time or now_ist()
        previous = previous_prices or {}
        blocking: list[str] = []
        warnings: list[str] = []
        validated: list[str] = []
        rejected: list[str] = []
        ages: list[float] = []

        snapshot_age = self._age_seconds(now, snapshot.timestamp)
        ages.append(snapshot_age)
        if snapshot_age < 0:
            blocking.append("snapshot timestamp is in the future")
        elif snapshot_age > self.policy.max_option_chain_age_seconds:
            blocking.append("snapshot is stale")

        self._validate_base_quote(
            name="spot", token=snapshot.spot_instrument_token, price=snapshot.spot_price,
            timestamp=snapshot.spot_timestamp, now=now, previous=previous,
            blocking=blocking, warnings=warnings, validated=validated, rejected=rejected, ages=ages,
        )
        if self.policy.require_futures or snapshot.futures_price is not None or snapshot.futures_timestamp is not None:
            if snapshot.futures_price is None or snapshot.futures_timestamp is None or not snapshot.futures_instrument_token:
                blocking.append("configured futures quote is incomplete")
            else:
                self._validate_base_quote(
                    name="futures", token=snapshot.futures_instrument_token, price=snapshot.futures_price,
                    timestamp=snapshot.futures_timestamp, now=now, previous=previous,
                    blocking=blocking, warnings=warnings, validated=validated, rejected=rejected, ages=ages,
                )

        chain = snapshot.option_chain
        chain_age = self._age_seconds(now, chain.timestamp)
        ages.append(chain_age)
        if chain_age < 0:
            blocking.append("option-chain timestamp is in the future")
        elif chain_age > self.policy.max_option_chain_age_seconds:
            blocking.append("option chain is stale")
        if chain.underlying.upper() != snapshot.underlying.upper():
            blocking.append("option-chain underlying does not match spot")
        if chain.spot_price <= 0:
            blocking.append("option-chain spot price must be positive")

        identities: set[tuple[object, ...]] = set()
        valid_identities: set[tuple[object, ...]] = set()
        safest_quote_available = snapshot.spot_price > 0
        for contract in chain.contracts:
            identity = (contract.expiry, float(contract.strike), contract.option_type.upper())
            token = self._token(contract)
            unique_identity = (token,) if token else identity
            reasons = self._contract_reasons(contract, chain.timestamp, snapshot.spot_timestamp, now, previous)
            quote_age = self._age_seconds(now, self._quote_timestamp(contract, chain.timestamp))
            ages.append(quote_age)
            if unique_identity in identities:
                reasons.append("duplicate option contract")
            identities.add(unique_identity)
            label = token or contract.symbol or repr(identity)
            if reasons:
                rejected.append(label)
                blocking.extend(f"{label}: {reason}" for reason in reasons)
            else:
                validated.append(label)
                valid_identities.add(identity)
                safest_quote_available = True

        for leg in required_legs:
            identity = (leg.expiry, float(leg.strike), leg.option_type.upper())
            if identity not in valid_identities:
                blocking.append(
                    f"missing required strategy leg {leg.expiry.isoformat()} {leg.strike:g} {leg.option_type.upper()}"
                )

        blocking = list(dict.fromkeys(blocking))
        warnings = list(dict.fromkeys(warnings))
        entry_allowed = not blocking
        exit_allowed = entry_allowed or (emergency_exit and safest_quote_available)
        if emergency_exit and not entry_allowed and exit_allowed:
            warnings.append("entry blocked; emergency exit may use the safest validated available quote")
        return MarketSnapshotValidationResult(
            valid=not blocking,
            entry_allowed=entry_allowed,
            exit_allowed=exit_allowed,
            blocking_reasons=tuple(blocking),
            warnings=tuple(warnings),
            snapshot_timestamp=snapshot.timestamp,
            oldest_quote_age=max(ages, default=0.0),
            validated_instruments=tuple(validated),
            rejected_instruments=tuple(rejected),
        )

    def _validate_base_quote(self, *, name: str, token: str, price: float, timestamp: datetime,
                             now: datetime, previous: Mapping[str, float], blocking: list[str],
                             warnings: list[str], validated: list[str], rejected: list[str],
                             ages: list[float]) -> None:
        reasons: list[str] = []
        if not token:
            reasons.append("instrument token is missing")
        if price <= 0:
            reasons.append("price must be positive")
        age = self._age_seconds(now, timestamp)
        ages.append(age)
        if age < 0:
            reasons.append("timestamp is in the future")
        elif age > self.policy.max_quote_age_seconds:
            reasons.append("quote is stale")
        self._append_jump_reason(token, price, previous, reasons)
        if reasons:
            rejected.append(token or name)
            blocking.extend(f"{name}: {reason}" for reason in reasons)
        else:
            validated.append(token)

    def _contract_reasons(self, contract: OptionContract, chain_timestamp: datetime,
                          spot_timestamp: datetime, now: datetime,
                          previous: Mapping[str, float]) -> list[str]:
        reasons: list[str] = []
        token = self._token(contract)
        timestamp = self._quote_timestamp(contract, chain_timestamp)
        if not token:
            reasons.append("instrument token is missing")
        if contract.expiry is None:
            reasons.append("expiry is missing")
        if contract.strike <= 0:
            reasons.append("strike must be positive")
        if contract.option_type.upper() not in {"CE", "PE"}:
            reasons.append("option type must be CE or PE")
        if contract.lot_size <= 0:
            reasons.append("lot size must be positive")
        if contract.available_quantity is None or contract.available_quantity <= 0:
            reasons.append("available quantity must be positive")
        if contract.bid is None or contract.ask is None:
            reasons.append("bid and ask are required")
        else:
            if contract.bid <= 0 or contract.ask <= 0:
                reasons.append("bid and ask must be positive")
            if contract.bid > contract.ask:
                reasons.append("bid exceeds ask")
        age = self._age_seconds(now, timestamp)
        if age < 0:
            reasons.append("quote timestamp is in the future")
        elif age > self.policy.max_quote_age_seconds:
            reasons.append("quote is stale")
        if abs((timestamp - spot_timestamp).total_seconds()) > self.policy.max_underlying_option_skew_seconds:
            reasons.append("underlying and option timestamps are misaligned")
        reference_price = contract.last_price or contract.ask or contract.bid or 0.0
        if reference_price <= 0:
            reasons.append("option price must be positive")
        self._append_jump_reason(token, reference_price, previous, reasons)
        return reasons

    def _append_jump_reason(self, token: str, price: float, previous: Mapping[str, float], reasons: list[str]) -> None:
        prior = previous.get(token)
        if prior is not None and prior > 0 and price > 0:
            jump = abs(price / prior - 1.0) * 100.0
            if jump > self.policy.max_price_jump_percent:
                reasons.append(f"abnormal price jump {jump:.2f}%")

    @staticmethod
    def _token(contract: OptionContract) -> str:
        return str(contract.instrument_token or contract.metadata.get("token") or "")

    @staticmethod
    def _quote_timestamp(contract: OptionContract, fallback: datetime) -> datetime:
        value = contract.quote_timestamp or contract.metadata.get("quote_timestamp") or fallback
        return datetime.fromisoformat(value) if isinstance(value, str) else value

    @staticmethod
    def _age_seconds(now: datetime, timestamp: datetime) -> float:
        try:
            return (now - timestamp).total_seconds()
        except TypeError:
            return float("inf")

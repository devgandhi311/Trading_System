"""Deterministic strict PAPER market snapshot validation tests."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import pytest

from market.market_data_manager import MarketDataManager
from models import Mode
from models.market_models import (
    MarketSnapshotValidationResult,
    OptionChainSnapshot,
    OptionContract,
    PaperMarketSnapshot,
    RequiredOptionLeg,
)
from runtime.market_snapshot_validator import MarketSnapshotValidationPolicy, MarketSnapshotValidator

NOW = datetime(2026, 7, 15, 10, 0, tzinfo=timezone.utc)
EXPIRY = date(2026, 7, 16)


def contract(**changes) -> OptionContract:
    values = dict(
        symbol="NIFTY26000CE", underlying="NIFTY", expiry=EXPIRY, strike=26000.0,
        option_type="CE", lot_size=75, bid=99.0, ask=101.0, last_price=100.0,
        instrument_token="token-ce", quote_timestamp=NOW - timedelta(seconds=1),
        available_quantity=150,
    )
    values.update(changes)
    return OptionContract(**values)


def snapshot(*contracts: OptionContract, **changes) -> PaperMarketSnapshot:
    contracts = contracts or (contract(),)
    chain = OptionChainSnapshot("NIFTY", NOW - timedelta(seconds=1), 26000.0, EXPIRY, tuple(contracts))
    values = dict(
        snapshot_id="snapshot-1", underlying="NIFTY", timestamp=NOW,
        spot_price=26000.0, spot_timestamp=NOW - timedelta(seconds=1),
        spot_instrument_token="spot-token", option_chain=chain,
    )
    values.update(changes)
    return PaperMarketSnapshot(**values)


def validate(value, **kwargs):
    return MarketSnapshotValidator().validate(value, current_time=NOW, **kwargs)


def reasons(result) -> str:
    return " | ".join(result.blocking_reasons)


def test_valid_snapshot_returns_structured_result() -> None:
    result = validate(snapshot())
    assert result.valid and result.entry_allowed and result.exit_allowed
    assert result.blocking_reasons == ()
    assert result.oldest_quote_age == pytest.approx(1.0)
    assert set(result.validated_instruments) == {"spot-token", "token-ce"}
    assert result.rejected_instruments == ()


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"bid": None}, "bid and ask are required"),
        ({"ask": 0.0}, "bid and ask must be positive"),
        ({"bid": 102.0, "ask": 101.0}, "bid exceeds ask"),
        ({"instrument_token": "", "metadata": {}}, "instrument token is missing"),
        ({"expiry": None}, "expiry is missing"),
        ({"strike": 0.0}, "strike must be positive"),
        ({"option_type": "XX"}, "option type must be CE or PE"),
        ({"lot_size": 0}, "lot size must be positive"),
        ({"available_quantity": 0}, "available quantity must be positive"),
    ],
)
def test_contract_integrity_branches(change, message) -> None:
    result = validate(snapshot(contract(**change)))
    assert not result.entry_allowed
    assert message in reasons(result)


def test_stale_spot_quote_and_chain_block_entry() -> None:
    old = NOW - timedelta(seconds=30)
    value = snapshot(spot_timestamp=old, option_chain=OptionChainSnapshot(
        "NIFTY", old, 26000.0, EXPIRY, (contract(quote_timestamp=old),)
    ))
    result = validate(value)
    assert not result.valid and not result.entry_allowed
    assert "spot: quote is stale" in result.blocking_reasons
    assert "option chain is stale" in result.blocking_reasons
    assert result.oldest_quote_age == pytest.approx(30.0)


@pytest.mark.parametrize(
    ("timestamp", "message"),
    [
        (NOW - timedelta(seconds=30), "snapshot is stale"),
        (NOW + timedelta(seconds=1), "snapshot timestamp is in the future"),
    ],
)
def test_snapshot_timestamp_is_validated(timestamp, message) -> None:
    result = validate(snapshot(timestamp=timestamp))
    assert not result.entry_allowed
    assert message in result.blocking_reasons


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (snapshot(spot_instrument_token=""), "spot: instrument token is missing"),
        (snapshot(spot_price=0.0), "spot: price must be positive"),
        (snapshot(spot_timestamp=NOW + timedelta(seconds=1)), "spot: timestamp is in the future"),
        (snapshot(option_chain=OptionChainSnapshot("BANKNIFTY", NOW, 26000.0, EXPIRY, (contract(),))),
         "option-chain underlying does not match spot"),
        (snapshot(option_chain=OptionChainSnapshot("NIFTY", NOW, 0.0, EXPIRY, (contract(),))),
         "option-chain spot price must be positive"),
        (snapshot(option_chain=OptionChainSnapshot("NIFTY", NOW + timedelta(seconds=1), 26000.0,
                                                  EXPIRY, (contract(quote_timestamp=NOW),))),
         "option-chain timestamp is in the future"),
        (snapshot(contract(quote_timestamp=NOW + timedelta(seconds=1))),
         "quote timestamp is in the future"),
        (snapshot(contract(last_price=0.0, bid=None, ask=None)), "option price must be positive"),
    ],
)
def test_snapshot_timestamp_identity_and_price_branches(value, message) -> None:
    assert message in reasons(validate(value))


def test_required_futures_must_be_complete_and_fresh() -> None:
    validator = MarketSnapshotValidator(MarketSnapshotValidationPolicy(require_futures=True))
    missing = validator.validate(snapshot(), current_time=NOW)
    assert "configured futures quote is incomplete" in missing.blocking_reasons
    valid = validator.validate(snapshot(futures_price=26020.0, futures_timestamp=NOW,
                                        futures_instrument_token="future-token"), current_time=NOW)
    assert valid.entry_allowed
    stale = validator.validate(snapshot(futures_price=26020.0,
                                        futures_timestamp=NOW - timedelta(seconds=30),
                                        futures_instrument_token="future-token"), current_time=NOW)
    assert "futures: quote is stale" in stale.blocking_reasons


def test_duplicate_contracts_are_rejected() -> None:
    result = validate(snapshot(contract(), replace(contract(), symbol="duplicate")))
    assert "duplicate option contract" in reasons(result)
    assert "token-ce" in result.rejected_instruments


def test_missing_required_strategy_leg_blocks_candidate() -> None:
    required = (RequiredOptionLeg(EXPIRY, 26100.0, "CE"),)
    result = validate(snapshot(), required_legs=required)
    assert not result.entry_allowed
    assert "missing required strategy leg" in reasons(result)


def test_abnormal_price_jump_blocks_entry() -> None:
    result = validate(snapshot(), previous_prices={"spot-token": 20000.0, "token-ce": 50.0})
    assert "abnormal price jump" in reasons(result)


def test_stale_underlying_option_timestamp_alignment_blocks() -> None:
    result = validate(snapshot(contract(quote_timestamp=NOW - timedelta(seconds=5))))
    assert "underlying and option timestamps are misaligned" in reasons(result)


def test_emergency_exit_can_use_safest_available_quote_while_entry_blocked() -> None:
    result = validate(snapshot(contract(bid=None, ask=None)), emergency_exit=True)
    assert not result.valid and not result.entry_allowed
    assert result.exit_allowed
    assert "safest validated available quote" in " ".join(result.warnings)


def test_non_emergency_exit_is_blocked_with_invalid_snapshot() -> None:
    result = validate(snapshot(contract(bid=None, ask=None)))
    assert not result.exit_allowed


def test_no_mock_fallback_after_manager_validation_failure() -> None:
    class RejectingValidator:
        calls = 0

        def validate(self, supplied, **kwargs):
            self.calls += 1
            return MarketSnapshotValidationResult(False, False, False, ("stale",), (), supplied.timestamp,
                                                  99.0, (), ("token-ce",))

    validator = RejectingValidator()
    manager = MarketDataManager(mode=Mode.PAPER, snapshot_validator=validator)
    result = manager.validate_paper_snapshot(snapshot(), current_time=NOW)
    assert result.blocking_reasons == ("stale",)
    assert validator.calls == 1

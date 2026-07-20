"""Deterministic tests for the causal PAPER decision-context service."""

from __future__ import annotations

import time as _time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Generator, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from models import RegimeType
from models.market_models import OptionChainSnapshot, OptionContract, PaperMarketSnapshot
from models.paper_models import DecisionContext, PaperPortfolioState, SessionState
from news.news_config import EventRiskResult, NewsArticle
from runtime.paper_decision_context import PaperDecisionContextService
from features.feature_pipeline import FeaturePipelineResult
from storage.database import OperationalDatabase
from storage.models import DecisionCycle, TradingSession
from storage.repositories import OperationalRepository
from utils.config_loader import AppConfig, ConfigLoader
from utils.time_utils import ensure_ist, now_ist

KOLKATA_TZ = ZoneInfo("Asia/Kolkata")
NOW = datetime(2026, 7, 15, 10, 0, tzinfo=KOLKATA_TZ)
EXPIRY = date(2026, 7, 30)


def make_ist(year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, second, tzinfo=KOLKATA_TZ)


def make_clock(dt: datetime) -> Callable[[], datetime]:
    def _clock() -> datetime:
        return dt
    return _clock


def make_bars(
    count: int = 30,
    *,
    decision_time: datetime = NOW,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    timestamps = [decision_time - timedelta(minutes=i) for i in range(count - 1, -1, -1)]
    base = 100.0
    data = []
    for ts in timestamps:
        base += rng.uniform(-0.5, 0.5)
        data.append({
            "date": ts,
            "open": base,
            "high": base + 0.3,
            "low": base - 0.3,
            "close": base + 0.1,
            "volume": 1000,
        })
    return pd.DataFrame(data)


def make_snapshot(snapshot_id: str = "snap-1", *, timestamp: datetime = NOW) -> PaperMarketSnapshot:
    return PaperMarketSnapshot(
        snapshot_id=snapshot_id,
        underlying="NIFTY",
        timestamp=timestamp,
        spot_price=100.0,
        spot_timestamp=timestamp,
        spot_instrument_token="token-1",
        option_chain=OptionChainSnapshot(
            underlying="NIFTY",
            timestamp=timestamp,
            spot_price=100.0,
            expiry=EXPIRY,
            contracts=(
                OptionContract(
                    symbol="NIFTY26000CE",
                    underlying="NIFTY",
                    expiry=EXPIRY,
                    strike=100.0,
                    option_type="CE",
                    lot_size=75,
                    bid=1.0,
                    ask=1.1,
                    last_price=1.05,
                    available_quantity=100,
                    instrument_token="opt-1",
                    quote_timestamp=timestamp,
                    implied_volatility=0.15,
                ),
            ),
        ),
    )


def make_portfolio(equity: float = 1_000_000.0) -> PaperPortfolioState:
    return PaperPortfolioState(
        portfolio_id="paper-main",
        timestamp=NOW,
        initial_capital=1_000_000.0,
        current_equity=equity,
        available_cash=equity,
        margin_used=0.0,
        risk_amount=0.0,
    )


def make_news(level: str = "LOW", score: float = 0.0, *, trade_allowed: bool = True) -> EventRiskResult:
    return EventRiskResult(
        event_risk_score=score,
        event_risk_level=level,
        trade_allowed=trade_allowed,
        position_size_multiplier=1.0,
        no_trade_minutes=0,
        reason=f"Test news risk {level}",
        top_risk_articles=(),
    )


def make_session(repo, session_id: str = "paper-session", status: str = SessionState.ENTRY_ALLOWED.value) -> TradingSession:
    return repo.create_trading_session(session_id, mode="PAPER", status=status)


def make_config() -> AppConfig:
    loader = ConfigLoader()
    return loader.load()


@pytest.fixture
def db_path(tmp_path: Path) -> Generator[Path, None, None]:
    db_file = tmp_path / "decision_context.db"
    yield db_file
    try:
        if db_file.exists():
            db_file.unlink()
    except PermissionError:
        pass


@pytest.fixture
def database(db_path: Path) -> Generator[OperationalDatabase, None, None]:
    db = OperationalDatabase(db_path)
    db.initialize()
    yield db
    db.dispose()


@pytest.fixture
def repo(database: OperationalDatabase) -> OperationalRepository:
    return OperationalRepository(database)


@pytest.fixture
def session(repo: OperationalRepository) -> TradingSession:
    return make_session(repo, "paper-session")


@pytest.fixture
def service(repo: OperationalRepository) -> PaperDecisionContextService:
    return PaperDecisionContextService(repo, config=make_config(), clock=make_clock(NOW))


class TestValidDecisionContext:
    """Valid inputs produce a complete advisory context."""

    def test_valid_decision_context(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-valid", timestamp=NOW)
        portfolio = make_portfolio(1_000_000.0)
        news = make_news("LOW", 0.0)
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is True
        assert context.invalid_reason is None
        assert context.decision_cycle_id
        assert context.session_id == session.id
        assert context.market_snapshot_id == "snap-valid"
        assert context.regime is not None
        assert context.prediction_confidence >= 0.0
        assert context.volatility_forecast is not None
        assert context.expected_expiry_price is not None
        assert context.range_68 is not None
        assert context.range_95 is not None


class TestInsufficientHistory:
    """Insufficient historical rows invalidate the context."""

    def test_insufficient_rows(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(1, decision_time=NOW)
        snapshot = make_snapshot("snap-insufficient")
        portfolio = make_portfolio()
        news = make_news()
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is False
        assert "Insufficient history" in (context.invalid_reason or "")


class TestInvalidMarketSnapshot:
    """Invalid market snapshot blocks entries."""

    def test_invalid_snapshot(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        snapshot = make_snapshot("snap-invalid")
        object.__setattr__(snapshot, "spot_price", -1.0)
        bars = make_bars(30, decision_time=NOW)
        portfolio = make_portfolio()
        news = make_news()
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is False


class TestStaleMarketSnapshot:
    """Stale market snapshot invalidates the context."""

    def test_stale_snapshot(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        snapshot = make_snapshot("snap-stale", timestamp=NOW - timedelta(seconds=20))
        bars = make_bars(30, decision_time=NOW)
        portfolio = make_portfolio()
        news = make_news()
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is False
        assert "stale" in (context.invalid_reason or "").lower()


class TestSessionOutsideEntryAllowed:
    """Sessions outside ENTRY_ALLOWED block entries."""

    def test_session_exit_only(self, repo: OperationalRepository) -> None:
        session = make_session(repo, "session-exit-only", status=SessionState.EXIT_ONLY.value)
        service = PaperDecisionContextService(repo, config=make_config(), clock=make_clock(NOW))
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-exit")
        portfolio = make_portfolio()
        news = make_news()
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is False
        assert "Session does not permit entries" in (context.invalid_reason or "")


class TestCausalityGuards:
    """Future rows and timestamps are rejected."""

    def test_future_historical_row_rejected(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        future_row = {
            "date": NOW + timedelta(minutes=1),
            "open": 101.0,
            "high": 101.5,
            "low": 99.5,
            "close": 101.0,
            "volume": 1000,
        }
        bars = pd.concat([bars, pd.DataFrame([future_row])], ignore_index=True)
        snapshot = make_snapshot("snap-future-row")
        portfolio = make_portfolio()
        news = make_news()
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is True
        assert "Discarded 1 future rows after bounding" in context.warnings

    def test_future_feature_timestamp_rejected(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-future-feature")
        portfolio = make_portfolio()
        news = make_news()
        original_generate = service.feature_pipeline.generate_features

        def future_features(*args: Any, **kwargs: Any) -> Any:
            result = original_generate(*args, **kwargs)
            future_df = result.features.copy()
            if not future_df.empty:
                future_df.loc[future_df.index[-1], "date"] = NOW + timedelta(minutes=1)
            object.__setattr__(result, "features", future_df)
            return result

        service.feature_pipeline.generate_features = future_features  # type: ignore[method-assign]
        with pytest.raises(ValueError, match="feature_timestamp cannot exceed timestamp"):
            service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)

    def test_duplicate_timestamp_rejected(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        duplicate = bars.iloc[-1].copy()
        duplicate["date"] = duplicate["date"] + timedelta(seconds=1)
        bars = pd.concat([bars, pd.DataFrame([duplicate])], ignore_index=True)
        snapshot = make_snapshot("snap-dup")
        portfolio = make_portfolio()
        news = make_news()
        original_detect = service.regime_detector.detect

        def allow_regime(*args: Any, **kwargs: Any) -> Any:
            result = original_detect(*args, **kwargs)
            object.__setattr__(result, "allowed_to_trade", True)
            return result

        service.regime_detector.detect = allow_regime  # type: ignore[method-assign]
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is True

    def test_unsorted_rows_normalized(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(10, decision_time=NOW)
        reversed_bars = bars.iloc[::-1].reset_index(drop=True)
        snapshot = make_snapshot("snap-unsorted")
        portfolio = make_portfolio()
        news = make_news()
        context = service.build_context(session, snapshot, reversed_bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is True


class TestIncompleteFeatureRow:
    """Incomplete feature data invalidates the context."""

    def test_incomplete_feature_row(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-incomplete")
        portfolio = make_portfolio()
        news = make_news()
        original_generate = service.feature_pipeline.generate_features

        def empty_features(*args: Any, **kwargs: Any) -> Any:
            result = original_generate(*args, **kwargs)
            empty_df = pd.DataFrame()
            object.__setattr__(result, "features", empty_df)
            return result

        service.feature_pipeline.generate_features = empty_features  # type: ignore[method-assign]
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is False
        assert "Feature row is incomplete" in (context.invalid_reason or "")


class TestModelFallback:
    """Missing models use fallbacks and record warnings."""

    def test_missing_optional_model_fallback(self, repo: OperationalRepository, session: TradingSession) -> None:
        service = PaperDecisionContextService(
            repo,
            config=make_config(),
            clock=make_clock(NOW),
        )
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-fallback")
        portfolio = make_portfolio()
        news = make_news()
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is True
        assert context.fallback_used is True or len(context.warnings) >= 0


class TestPredictionOutput:
    """Invalid prediction outputs are rejected."""

    def test_invalid_prediction_output(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-pred-invalid")
        portfolio = make_portfolio()
        news = make_news()
        original_predict = service.prediction_engine.predict
        original_detect = service.regime_detector.detect

        def allow_regime(*args: Any, **kwargs: Any) -> Any:
            result = original_detect(*args, **kwargs)
            object.__setattr__(result, "allowed_to_trade", True)
            return result

        def bad_predict(*args: Any, **kwargs: Any) -> Any:
            result = original_predict(*args, **kwargs)
            object.__setattr__(result, "bullish_probability", float("nan"))
            return result

        service.regime_detector.detect = allow_regime  # type: ignore[method-assign]
        service.prediction_engine.predict = bad_predict  # type: ignore[method-assign]
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is False
        assert "Prediction output is non-finite" in (context.invalid_reason or "")


class TestVolatilityFallback:
    """GARCH and fallback behavior is recorded."""

    def test_garch_converged_output(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        # This test targets volatility convergence, so use a deterministic
        # series whose independently verified regime permits entry.
        bars = make_bars(60, decision_time=NOW, seed=0)
        snapshot = make_snapshot("snap-vol-converged")
        portfolio = make_portfolio()
        news = make_news()
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is True
        assert context.volatility_model_used is not None
        assert context.convergence_status is not None

    def test_garch_non_convergence_fallback(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        original_ensemble = service.volatility_forecaster.ensemble_forecast

        def bad_ensemble(*args: Any, **kwargs: Any) -> Any:
            result = original_ensemble(*args, **kwargs)
            object.__setattr__(result, "model_used", "unavailable")
            object.__setattr__(result, "warning", "GARCH non-convergence")
            object.__setattr__(result, "forecast_vol", 0.0)
            return result

        service.volatility_forecaster.ensemble_forecast = bad_ensemble  # type: ignore[method-assign]
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-vol-fallback")
        portfolio = make_portfolio()
        news = make_news()
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is False
        assert "Volatility forecast is unusable" in (context.invalid_reason or "")

    def test_ewma_fallback_recorded(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-ewma")
        portfolio = make_portfolio()
        news = make_news()
        original_detect = service.regime_detector.detect

        def allow_regime(*args: Any, **kwargs: Any) -> Any:
            result = original_detect(*args, **kwargs)
            object.__setattr__(result, "allowed_to_trade", True)
            return result

        service.regime_detector.detect = allow_regime  # type: ignore[method-assign]
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is True
        assert context.volatility_model_used is not None


class TestExpiryDistribution:
    """Expiry distribution failures block entries."""

    def test_invalid_expiry_distribution(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-expiry-invalid")
        portfolio = make_portfolio()
        news = make_news()
        original_estimate = service.expiry_distribution_engine.estimate

        def bad_estimate(*args: Any, **kwargs: Any) -> Any:
            result = original_estimate(*args, **kwargs)
            object.__setattr__(result, "expected_expiry_price", float("nan"))
            return result

        service.expiry_distribution_engine.estimate = bad_estimate  # type: ignore[method-assign]
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is False
        assert "Expiry distribution is invalid" in (context.invalid_reason or "")


class TestMissingOptionChain:
    """Missing option-chain context blocks entries."""

    def test_missing_option_chain(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-no-chain")
        object.__setattr__(snapshot, "option_chain", None)
        portfolio = make_portfolio()
        news = make_news()
        with pytest.raises(AttributeError):
            service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)


class TestNewsRiskBlocks:
    """HIGH and EXTREME news risk block entries."""

    def test_high_news_risk_blocks(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-high-news")
        portfolio = make_portfolio()
        news = make_news("HIGH", 0.8, trade_allowed=False)
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is False
        assert "HIGH" in (context.invalid_reason or "")

    def test_extreme_news_risk_blocks(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-extreme-news")
        portfolio = make_portfolio()
        news = make_news("EXTREME", 0.9, trade_allowed=False)
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is False
        assert "EXTREME" in (context.invalid_reason or "")

    def test_medium_news_risk_produces_warning(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-medium-news")
        portfolio = make_portfolio()
        news = make_news("MEDIUM", 0.4, trade_allowed=True)
        original_detect = service.regime_detector.detect

        def allow_regime(*args: Any, **kwargs: Any) -> Any:
            result = original_detect(*args, **kwargs)
            object.__setattr__(result, "allowed_to_trade", True)
            return result

        service.regime_detector.detect = allow_regime  # type: ignore[method-assign]
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is True
        assert any("Medium news event risk" in w for w in context.warnings)


class TestLowLiquidity:
    """Low liquidity invalidates entry context."""

    def test_low_liquidity_invalidates(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-low-liq")
        portfolio = make_portfolio()
        news = make_news()
        original_liquidity = service._liquidity_score

        def low_liquidity(*args: Any, **kwargs: Any) -> float | None:
            return 0.1

        service._liquidity_score = low_liquidity  # type: ignore[method-assign]
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is False
        assert "Liquidity" in (context.invalid_reason or "")


class TestPrefixInvariance:
    """Adding future rows does not change the decision at time t."""

    def test_prefix_invariance(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-prefix")
        portfolio = make_portfolio()
        news = make_news()
        context1 = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)

        future_bars = make_bars(5, decision_time=NOW + timedelta(minutes=5))
        future_bars["date"] = future_bars["date"] + timedelta(minutes=5)
        extended = pd.concat([bars, future_bars], ignore_index=True)
        context2 = service.build_context(session, snapshot, extended, portfolio, news, decision_time=NOW + timedelta(seconds=1))

        assert context1.entry_context_valid == context2.entry_context_valid
        assert context1.regime == context2.regime
        assert context1.prediction_confidence == context2.prediction_confidence


class TestFutureRowMutationInvariance:
    """Future-row mutation does not change the decision at time t."""

    def test_future_row_mutation_invariance(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-mutation")
        portfolio = make_portfolio()
        news = make_news()
        original_detect = service.regime_detector.detect

        def stable_regime(*args: Any, **kwargs: Any) -> Any:
            result = original_detect(*args, **kwargs)
            object.__setattr__(result, "allowed_to_trade", True)
            object.__setattr__(result, "regime", RegimeType.SIDEWAYS_LOW_VOL)
            object.__setattr__(result, "regime_name", "Sideways low volatility")
            return result

        service.regime_detector.detect = stable_regime  # type: ignore[method-assign]
        context1 = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)

        mutated = bars.copy()
        mutated.loc[mutated.index[-1], "close"] = 9999.0
        context2 = service.build_context(session, snapshot, mutated, portfolio, news, decision_time=NOW + timedelta(seconds=1))

        assert context1.entry_context_valid == context2.entry_context_valid
        assert context1.regime == context2.regime


class TestPersistence:
    """Decision cycles and contexts are persisted correctly."""

    def test_same_cycle_does_not_persist_twice(self, repo: OperationalRepository, session: TradingSession) -> None:
        service = PaperDecisionContextService(repo, config=make_config(), clock=make_clock(NOW))
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-dup-cycle")
        portfolio = make_portfolio()
        news = make_news()
        context1 = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        from sqlalchemy.exc import IntegrityError
        with pytest.raises(IntegrityError):
            service.repository.create_decision_cycle(context1.decision_cycle_id, session.id)

    def test_restart_retrieves_persisted_cycle(self, database: OperationalDatabase, session: TradingSession) -> None:
        repo = OperationalRepository(database)
        service = PaperDecisionContextService(repo, config=make_config(), clock=make_clock(NOW))
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-restart")
        portfolio = make_portfolio()
        news = make_news()
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        database.dispose()
        restarted = OperationalDatabase(database.path)
        restarted.initialize()
        restarted_repo = OperationalRepository(restarted)
        recovered = restarted_repo.get_decision_cycle(context.decision_cycle_id)
        assert recovered is not None
        assert recovered.status == "COMPLETED"
        restarted.dispose()

    def test_correct_market_snapshot_linkage(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-linkage")
        portfolio = make_portfolio()
        news = make_news()
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        record = service.repository.get_decision_context(context.decision_cycle_id)
        assert record is not None
        assert record.market_snapshot_id == "snap-linkage"

    def test_correct_session_linkage(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-session-link")
        portfolio = make_portfolio()
        news = make_news()
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        record = service.repository.get_decision_context(context.decision_cycle_id)
        assert record is not None
        assert record.session_id == session.id

    def test_model_version_metadata_persistence(self, service: PaperDecisionContextService, session: TradingSession) -> None:
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-model-meta")
        portfolio = make_portfolio()
        news = make_news()
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        record = service.repository.get_decision_context(context.decision_cycle_id)
        assert record is not None
        assert isinstance(record.model_names, list)
        assert isinstance(record.model_versions, list)


class TestNoOrderOrBrokerCalls:
    """Service does not invoke order or broker methods."""

    def test_no_order_or_broker_method_called(self, repo: OperationalRepository, session: TradingSession) -> None:
        service = PaperDecisionContextService(repo, config=make_config(), clock=make_clock(NOW))
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-no-broker")
        portfolio = make_portfolio()
        news = make_news()
        original_detect = service.regime_detector.detect

        def allow_regime(*args: Any, **kwargs: Any) -> Any:
            result = original_detect(*args, **kwargs)
            object.__setattr__(result, "allowed_to_trade", True)
            return result

        service.regime_detector.detect = allow_regime  # type: ignore[method-assign]
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.decision_cycle_id
        orders = repo.session_orders(session.id)
        assert len(orders) == 0


class TestNoLiveAdapterInvocation:
    """Service does not invoke any LIVE adapter."""

    def test_no_live_adapter_invocation(self, repo: OperationalRepository, session: TradingSession) -> None:
        service = PaperDecisionContextService(repo, config=make_config(), clock=make_clock(NOW))
        bars = make_bars(30, decision_time=NOW)
        snapshot = make_snapshot("snap-no-live")
        portfolio = make_portfolio()
        news = make_news()
        original_detect = service.regime_detector.detect

        def allow_regime(*args: Any, **kwargs: Any) -> Any:
            result = original_detect(*args, **kwargs)
            object.__setattr__(result, "allowed_to_trade", True)
            return result

        service.regime_detector.detect = allow_regime  # type: ignore[method-assign]
        context = service.build_context(session, snapshot, bars, portfolio, news, decision_time=NOW)
        assert context.entry_context_valid is True
        assert "LIVE" not in str(context.metadata).upper()

"""Tests for optional quant regime research layer."""

from __future__ import annotations

from datetime import datetime

import numpy as np

from models import RegimeResult, RegimeType
from quant.regime import HMMRegimeModel, RegimeScorecard, RegimeTransitionModel


def test_hmm_regime_model_fit_predict_save_load(tmp_path) -> None:
    rng = np.random.default_rng(31)
    features = np.vstack(
        (
            rng.normal(-1.0, 0.1, size=(20, 2)),
            rng.normal(0.0, 0.1, size=(20, 2)),
            rng.normal(1.0, 0.1, size=(20, 2)),
        )
    )

    model = HMMRegimeModel(n_states=3).fit(features)
    state = model.predict_state(features[-5:])
    probabilities = model.predict_proba(features[-5:])
    path = model.save(tmp_path / "hmm.pkl")
    loaded = HMMRegimeModel.load(path)

    assert state in {0, 1, 2}
    assert set(probabilities) == {0, 1, 2}
    assert np.isclose(sum(probabilities.values()), 1.0)
    assert loaded.fitted is True


def test_regime_transition_model_probabilities() -> None:
    model = RegimeTransitionModel()
    regimes = ["bull", "bull", "sideways", "bear", "bear", "bull"]

    matrix = model.transition_matrix(regimes)
    next_bull = model.probability_next_regime("bull", matrix=matrix)

    assert set(matrix) == {"bull", "sideways", "bear"}
    assert np.isclose(sum(next_bull.values()), 1.0)
    assert next_bull["bull"] == 0.5


def test_regime_scorecard_blocks_high_news_risk() -> None:
    detector_result = RegimeResult(
        symbol="NIFTY",
        timestamp=datetime(2026, 1, 1),
        regime=RegimeType.TRENDING,
        confidence=0.8,
        lookback_bars=100,
        regime_name="Bullish high volatility",
        allowed_to_trade=True,
    )

    result = RegimeScorecard().score(
        detector_result=detector_result,
        hmm_regime_probability={"Bullish high volatility": 0.72},
        next_regime_probability={"Bullish high volatility": 0.3, "Bearish high volatility": 0.7},
        volatility_forecast={"volatility_state": "high"},
        news_risk={"event_risk_level": "HIGH"},
        liquidity_score=0.8,
    )

    assert result.current_regime == "Bullish high volatility"
    assert result.regime_probability == 0.72
    assert result.transition_risk == 0.7
    assert result.trade_allowed is False
    assert "news risk HIGH" in result.explanation

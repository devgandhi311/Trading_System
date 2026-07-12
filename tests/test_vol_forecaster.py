"""Tests for quant volatility forecaster."""

from __future__ import annotations

import numpy as np

from quant.volatility import VolatilityForecaster, realized_volatility


def _returns(size: int = 120) -> np.ndarray:
    rng = np.random.default_rng(23)
    return rng.normal(loc=0.0002, scale=0.012, size=size)


def test_realized_vol_forecast_returns_positive_annualized_vol() -> None:
    forecaster = VolatilityForecaster(realized_window=20)

    result = forecaster.realized_vol_forecast(_returns(), horizon_days=3)

    assert result.forecast_vol > 0
    assert result.forecast_vol_percent == result.forecast_vol * 100.0
    assert result.horizon_days == 3
    assert result.model_used == "realized_vol"
    assert result.volatility_state in {"low", "normal", "high"}


def test_garch_and_egarch_fallback_when_disabled() -> None:
    forecaster = VolatilityForecaster(garch_enabled=False, egarch_enabled=False)

    garch = forecaster.garch_forecast(_returns(), horizon_days=5)
    egarch = forecaster.egarch_forecast(_returns(), horizon_days=5)

    assert garch.model_used == "fallback_realized_vol"
    assert egarch.model_used == "fallback_realized_vol"
    assert "disabled" in str(garch.warning)
    assert garch.forecast_vol > 0


def test_ensemble_forecast_includes_standard_horizons() -> None:
    forecaster = VolatilityForecaster(garch_enabled=False, egarch_enabled=False)

    result = forecaster.ensemble_forecast(_returns(), days_to_expiry=9)

    assert result.forecast_vol > 0
    assert result.horizon_days == 9
    assert {1, 3, 5, 9}.issubset(result.forecasts_by_horizon)
    assert result.model_used.startswith("ensemble:")


def test_realized_volatility_handles_percent_inputs() -> None:
    decimal = realized_volatility([0.01, -0.01, 0.02, -0.015], window=4)
    percent = realized_volatility([1.0, -1.0, 2.0, -1.5], window=4)

    assert np.isclose(decimal, percent)

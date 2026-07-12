"""Volatility forecasting engine with realized and ARCH-family fallbacks."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from quant.volatility.garch_model import forecast_arch_volatility, forecast_volatility_hierarchy
from quant.volatility.volatility_features import realized_volatility, volatility_state


@dataclass(frozen=True)
class VolForecastResult:
    """Volatility forecast output."""

    forecast_vol: float
    forecast_vol_percent: float
    volatility_state: str
    confidence: float
    model_used: str
    warning: str | None = None
    horizon_days: int = 1
    forecasts_by_horizon: dict[int, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


class VolatilityForecaster:
    """Forecast volatility for fixed horizons and option expiry horizons."""

    def __init__(
        self,
        *,
        realized_window: int = 20,
        garch_enabled: bool = True,
        egarch_enabled: bool = True,
        fallback_to_realized_vol: bool = True,
    ) -> None:
        """Create a volatility forecaster."""

        self.realized_window = realized_window
        self.garch_enabled = garch_enabled
        self.egarch_enabled = egarch_enabled
        self.fallback_to_realized_vol = fallback_to_realized_vol

    def realized_vol_forecast(
        self,
        returns: Sequence[float] | np.ndarray,
        *,
        horizon_days: int = 1,
    ) -> VolForecastResult:
        """Forecast volatility using recent realized volatility."""

        vol = realized_volatility(returns, window=self.realized_window, annualize=True)
        return self._result(
            vol=vol,
            horizon_days=horizon_days,
            returns=returns,
            confidence=0.55 if vol > 0 else 0.0,
            model_used="realized_vol",
            warning=None if vol > 0 else "Not enough returns for realized volatility.",
        )

    def garch_forecast(
        self,
        returns: Sequence[float] | np.ndarray,
        *,
        horizon_days: int = 1,
    ) -> VolForecastResult:
        """Forecast volatility with GARCH hierarchy, falling back safely."""

        if not self.garch_enabled:
            return self._fallback(returns, horizon_days, "GARCH disabled.")
        forecast = forecast_volatility_hierarchy(returns, horizon_days=horizon_days, preferred_model="GARCH")
        confidence = 0.70 if forecast.model_used == "garch" else 0.58 if forecast.model_used == "egarch" else 0.45
        return self._result(
            vol=forecast.volatility,
            horizon_days=horizon_days,
            returns=returns,
            confidence=confidence,
            model_used=forecast.model_used,
            warning=forecast.warning,
        )

    def egarch_forecast(
        self,
        returns: Sequence[float] | np.ndarray,
        *,
        horizon_days: int = 1,
    ) -> VolForecastResult:
        """Forecast volatility with EGARCH hierarchy, falling back safely."""

        if not self.egarch_enabled:
            return self._fallback(returns, horizon_days, "EGARCH disabled.")
        forecast = forecast_volatility_hierarchy(returns, horizon_days=horizon_days, preferred_model="EGARCH")
        confidence = 0.68 if forecast.model_used == "egarch" else 0.58 if forecast.model_used == "garch" else 0.45
        return self._result(
            vol=forecast.volatility,
            horizon_days=horizon_days,
            returns=returns,
            confidence=confidence,
            model_used=forecast.model_used,
            warning=forecast.warning,
        )

    def ensemble_forecast(
        self,
        returns: Sequence[float] | np.ndarray,
        *,
        days_to_expiry: int,
    ) -> VolForecastResult:
        """Blend realized volatility with one cached ARCH-hierarchy forecast per horizon."""

        horizons = tuple(dict.fromkeys((1, 3, 5, max(1, int(days_to_expiry)))))
        forecasts_by_horizon: dict[int, float] = {}
        warnings: list[str] = []
        used_models: set[str] = set()
        expiry_horizon = max(1, int(days_to_expiry))
        hierarchy_result = None
        if self.garch_enabled or self.egarch_enabled:
            preferred = "GARCH" if self.garch_enabled else "EGARCH"
            hierarchy_result = forecast_volatility_hierarchy(returns, horizon_days=expiry_horizon, preferred_model=preferred)
        for horizon in horizons:
            components = [self.realized_vol_forecast(returns, horizon_days=horizon)]
            if hierarchy_result is not None:
                confidence = 0.72 if hierarchy_result.model_used in {"garch", "egarch"} else 0.45
                components.append(
                    self._result(
                        vol=hierarchy_result.volatility,
                        horizon_days=horizon,
                        returns=returns,
                        confidence=confidence,
                        model_used=hierarchy_result.model_used,
                        warning=hierarchy_result.warning,
                    )
                )
            else:
                components.append(self._fallback(returns, horizon, "GARCH/EGARCH disabled."))
            valid = [item for item in components if item.forecast_vol > 0]
            if not valid:
                forecasts_by_horizon[horizon] = 0.0
                warnings.append(f"No valid volatility forecast for {horizon}d horizon.")
                continue
            forecasts_by_horizon[horizon] = float(
                np.average(
                    [item.forecast_vol for item in valid],
                    weights=[max(item.confidence, 0.01) for item in valid],
                )
            )
            used_models.update(item.model_used for item in valid)
            warnings.extend(item.warning for item in components if item.warning)

        expiry_horizon = max(1, int(days_to_expiry))
        forecast_vol = forecasts_by_horizon.get(expiry_horizon, forecasts_by_horizon[horizons[-1]])
        confidence = 0.75 if len(used_models) > 1 else 0.55
        return self._result(
            vol=forecast_vol,
            horizon_days=expiry_horizon,
            returns=returns,
            confidence=confidence,
            model_used="ensemble:" + "+".join(sorted(used_models)),
            warning="; ".join(dict.fromkeys(warnings)) or None,
            forecasts_by_horizon=forecasts_by_horizon,
        )

    def _fallback(self, returns: Sequence[float] | np.ndarray, horizon_days: int, warning: str) -> VolForecastResult:
        if not self.fallback_to_realized_vol:
            return self._result(
                vol=0.0,
                horizon_days=horizon_days,
                returns=returns,
                confidence=0.0,
                model_used="unavailable",
                warning=warning,
            )
        result = self.realized_vol_forecast(returns, horizon_days=horizon_days)
        return VolForecastResult(
            forecast_vol=result.forecast_vol,
            forecast_vol_percent=result.forecast_vol_percent,
            volatility_state=result.volatility_state,
            confidence=max(result.confidence * 0.8, 0.0),
            model_used="fallback_realized_vol",
            warning=warning,
            horizon_days=horizon_days,
            forecasts_by_horizon=result.forecasts_by_horizon,
            metadata=result.metadata,
        )

    @staticmethod
    def _result(
        *,
        vol: float,
        horizon_days: int,
        returns: Sequence[float] | np.ndarray,
        confidence: float,
        model_used: str,
        warning: str | None = None,
        forecasts_by_horizon: dict[int, float] | None = None,
    ) -> VolForecastResult:
        annual_vol = max(float(vol), 0.0)
        return VolForecastResult(
            forecast_vol=annual_vol,
            forecast_vol_percent=annual_vol * 100.0,
            volatility_state=volatility_state(annual_vol, returns),
            confidence=float(min(max(confidence, 0.0), 1.0)),
            model_used=model_used,
            warning=warning,
            horizon_days=max(1, int(horizon_days)),
            forecasts_by_horizon=forecasts_by_horizon or {max(1, int(horizon_days)): annual_vol},
            metadata={"realized_window": None},
        )

"""GARCH, EGARCH, EWMA, and historical volatility forecasts with caching."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from hashlib import blake2b
import logging
from math import sqrt
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from quant.volatility.volatility_features import TRADING_DAYS_PER_YEAR, clean_returns, realized_volatility

LOGGER = logging.getLogger(__name__)
_FORECAST_CACHE: dict[tuple[str, int, str], "VolatilityModelForecast"] = {}
_WARNING_CACHE: set[str] = set()
_PERFORMANCE_ROWS: list[dict[str, Any]] = []


@dataclass(frozen=True)
class VolatilityModelForecast:
    """Volatility forecast result with diagnostics."""

    volatility: float
    warning: str | None
    model_used: str
    fit_count: int
    convergence_failure: bool
    runtime_seconds: float
    cache_hit: bool = False
    forecast_accuracy_error: float | None = None


def forecast_arch_volatility(
    returns: Sequence[float] | np.ndarray,
    *,
    horizon_days: int,
    model: str = "GARCH",
) -> tuple[float | None, str | None]:
    """Forecast annualized volatility using the safe fallback hierarchy."""

    forecast = forecast_volatility_hierarchy(returns, horizon_days=horizon_days, preferred_model=model)
    warning = forecast.warning
    if forecast.model_used not in {"garch", "egarch"} and warning is None:
        warning = f"ARCH optimizer unavailable; used {forecast.model_used} fallback."
    return forecast.volatility, warning


def forecast_volatility_hierarchy(
    returns: Sequence[float] | np.ndarray,
    *,
    horizon_days: int,
    preferred_model: str = "GARCH",
) -> VolatilityModelForecast:
    """Forecast volatility once per window using GARCH -> EGARCH -> EWMA -> historical fallback."""

    cleaned = clean_returns(returns)
    started = time.perf_counter()
    horizon = max(1, int(horizon_days))
    cache_key = (_returns_key(cleaned), horizon, preferred_model.upper())
    if cache_key in _FORECAST_CACHE:
        cached = _FORECAST_CACHE[cache_key]
        result = VolatilityModelForecast(
            cached.volatility,
            cached.warning,
            cached.model_used,
            0,
            cached.convergence_failure,
            0.0,
            cache_hit=True,
            forecast_accuracy_error=cached.forecast_accuracy_error,
        )
        _record_performance(result, cleaned, horizon)
        return result

    if cleaned.size < 30:
        result = _fallback_result(cleaned, horizon, warning="Not enough returns for ARCH volatility model.", started=started)
        _FORECAST_CACHE[cache_key] = result
        _record_performance(result, cleaned, horizon)
        return result
    try:
        from arch import arch_model
    except ImportError:
        result = _fallback_result(cleaned, horizon, warning="arch package is not installed.", started=started)
        _FORECAST_CACHE[cache_key] = result
        _record_performance(result, cleaned, horizon)
        return result

    scaled_returns = cleaned * 100.0
    failures: list[str] = []
    fits = 0
    for model_name in _fit_order(preferred_model):
        fits += 1
        try:
            fitted_model = (
                arch_model(scaled_returns, mean="Constant", vol="EGARCH", p=1, o=1, q=1, rescale=False)
                if model_name == "EGARCH"
                else arch_model(scaled_returns, mean="Constant", vol="GARCH", p=1, q=1, rescale=False)
            )
            fitted = fitted_model.fit(disp="off", show_warning=False)
            forecast = fitted.forecast(horizon=horizon, reindex=False)
            variance = np.asarray(forecast.variance.iloc[-1], dtype=float)
            daily_vol_percent = float(np.sqrt(np.mean(variance)))
            volatility = daily_vol_percent / 100.0 * sqrt(TRADING_DAYS_PER_YEAR)
            result = VolatilityModelForecast(
                volatility=volatility,
                warning=None if not failures else "; ".join(failures),
                model_used=model_name.lower(),
                fit_count=fits,
                convergence_failure=bool(failures),
                runtime_seconds=time.perf_counter() - started,
                forecast_accuracy_error=_accuracy_proxy(cleaned, volatility),
            )
            _FORECAST_CACHE[cache_key] = result
            _record_performance(result, cleaned, horizon)
            return result
        except Exception as exc:  # pragma: no cover - exact arch failures vary by platform/data.
            message = f"{model_name} forecast failed: {exc}"
            failures.append(message)
            _warn_once(message, model_name)

    result = _fallback_result(cleaned, horizon, warning="; ".join(failures), started=started)
    _FORECAST_CACHE[cache_key] = result
    _record_performance(result, cleaned, horizon)
    return result


def save_garch_performance_report(path: str | Path = "reports/debug/garch_performance_report.xlsx") -> Path:
    """Save accumulated volatility forecast diagnostics."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = pd.DataFrame(_PERFORMANCE_ROWS)
    if rows.empty:
        rows = pd.DataFrame(
            [
                {
                    "model_used": "none",
                    "fit_count": 0,
                    "convergence_failure": False,
                    "runtime_seconds": 0.0,
                    "cache_hit": False,
                    "forecast_accuracy_error": None,
                    "horizon_days": None,
                    "observations": 0,
                    "warning": None,
                }
            ]
        )
    accuracy = pd.to_numeric(rows["forecast_accuracy_error"], errors="coerce").dropna()
    summary = pd.DataFrame(
        [
            {
                "number_of_fits": int(rows["fit_count"].sum()),
                "number_of_convergence_failures": int(rows["convergence_failure"].sum()),
                "fallback_usage": int((~rows["model_used"].isin(["garch", "egarch"])).sum()),
                "average_runtime": float(rows["runtime_seconds"].mean()),
                "forecast_accuracy": float(accuracy.mean()) if not accuracy.empty else None,
                "cache_hits": int(rows["cache_hit"].sum()),
                "calls": len(rows),
            }
        ]
    )
    fallback = rows.groupby("model_used", dropna=False).size().reset_index(name="count")
    with pd.ExcelWriter(output) as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        fallback.to_excel(writer, sheet_name="Fallback Usage", index=False)
        rows.to_excel(writer, sheet_name="Forecast Calls", index=False)
    return output


def reset_garch_performance_stats() -> None:
    """Clear in-memory performance diagnostics without clearing forecast cache."""

    _PERFORMANCE_ROWS.clear()


def _fit_order(preferred_model: str) -> tuple[str, ...]:
    preferred = preferred_model.upper()
    if preferred == "EGARCH":
        return ("EGARCH", "GARCH")
    return ("GARCH", "EGARCH")


def _fallback_result(cleaned: np.ndarray, horizon: int, *, warning: str, started: float) -> VolatilityModelForecast:
    ewma = _ewma_volatility(cleaned)
    if ewma > 0:
        return VolatilityModelForecast(
            ewma,
            warning,
            "ewma",
            0,
            bool(warning),
            time.perf_counter() - started,
            forecast_accuracy_error=_accuracy_proxy(cleaned, ewma),
        )
    historical = realized_volatility(cleaned, window=max(20, horizon), annualize=True)
    return VolatilityModelForecast(
        historical,
        warning,
        "rolling_historical_volatility",
        0,
        bool(warning),
        time.perf_counter() - started,
        forecast_accuracy_error=_accuracy_proxy(cleaned, historical),
    )


def _ewma_volatility(cleaned: np.ndarray, *, lambda_: float = 0.94) -> float:
    if cleaned.size < 2:
        return 0.0
    weights = np.power(lambda_, np.arange(cleaned.size - 1, -1, -1, dtype=float))
    weights = weights / weights.sum()
    mean = float(np.sum(weights * cleaned))
    variance = float(np.sum(weights * np.square(cleaned - mean)))
    return sqrt(max(variance, 0.0)) * sqrt(TRADING_DAYS_PER_YEAR)


def _returns_key(cleaned: np.ndarray) -> str:
    rounded = np.round(cleaned.astype(float), 10)
    return blake2b(rounded.tobytes(), digest_size=16).hexdigest()


def _warn_once(message: str, model: str) -> None:
    if message in _WARNING_CACHE:
        return
    _WARNING_CACHE.add(message)
    LOGGER.warning("ARCH volatility forecast failed", extra={"model": model, "reason": message})


def _accuracy_proxy(cleaned: np.ndarray, forecast_vol: float) -> float | None:
    if cleaned.size < 22 or forecast_vol <= 0:
        return None
    realized = realized_volatility(cleaned[-20:], window=20, annualize=True)
    return abs(float(forecast_vol) - float(realized))


def _record_performance(result: VolatilityModelForecast, cleaned: np.ndarray, horizon: int) -> None:
    _PERFORMANCE_ROWS.append(
        {
            "model_used": result.model_used,
            "fit_count": result.fit_count,
            "convergence_failure": result.convergence_failure,
            "runtime_seconds": result.runtime_seconds,
            "cache_hit": result.cache_hit,
            "forecast_accuracy_error": result.forecast_accuracy_error,
            "horizon_days": horizon,
            "observations": int(cleaned.size),
            "warning": result.warning,
        }
    )

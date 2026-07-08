"""Train volatility and range models."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any

import pandas as pd

from ml.dataset_builder import DatasetBuilder
from ml.model_registry import ModelRegistry
from ml.training_utils import TrainingResult, train_walk_forward_classifiers, train_walk_forward_regressors
from utils.file_utils import ensure_parent_dir


def train_volatility_model(
    dataset: pd.DataFrame | None = None,
    *,
    registry: ModelRegistry | None = None,
    train_window: int = 252,
    test_window: int = 63,
) -> dict[str, TrainingResult | str]:
    """Train volatility expansion classifier, expected-range regressor, and optional GARCH/EGARCH."""

    data = dataset if dataset is not None else DatasetBuilder().build_dataset().dataset
    registry = registry or ModelRegistry()
    results: dict[str, TrainingResult | str] = {
        "volatility_expansion": train_walk_forward_classifiers(
            data,
            target="target_volatility_expansion",
            model_name_prefix="volatility_expansion",
            registry=registry,
            train_window=train_window,
            test_window=test_window,
        ),
        "expected_range": train_walk_forward_regressors(
            data,
            target="target_expected_range",
            model_name_prefix="expected_range",
            registry=registry,
            train_window=train_window,
            test_window=test_window,
        ),
    }
    results["garch"] = _train_optional_arch(data, registry=registry, model_name="garch", vol="GARCH")
    results["egarch"] = _train_optional_arch(data, registry=registry, model_name="egarch", vol="EGARCH")
    return results


def _train_optional_arch(
    data: pd.DataFrame,
    *,
    registry: ModelRegistry,
    model_name: str,
    vol: str,
) -> str:
    try:
        from arch import arch_model
    except Exception:
        return f"Skipped {model_name}: arch package unavailable"
    returns = pd.to_numeric(data.get("target_next_day_return"), errors="coerce").dropna() * 100
    if len(returns) < 20:
        return f"Skipped {model_name}: insufficient observations"
    model = arch_model(returns, vol=vol, p=1, q=1, rescale=False)
    fitted = model.fit(disp="off")
    entry = registry.save_model(fitted, model_name=model_name, model_type=vol, target="target_next_day_return")
    return f"Saved {model_name}: {entry.artifact_path}"

"""Train regime detection models."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.dataset_builder import DatasetBuilder
from ml.model_registry import ModelRegistry
from ml.training_utils import TrainingResult, feature_columns, train_walk_forward_classifiers
from utils.file_utils import ensure_parent_dir


def train_regime_model(
    dataset: pd.DataFrame | None = None,
    *,
    registry: ModelRegistry | None = None,
    train_window: int = 252,
    test_window: int = 63,
) -> dict[str, TrainingResult | str]:
    """Train supervised regime classifier and optional HMM regime model."""

    data = dataset if dataset is not None else DatasetBuilder().build_dataset().dataset
    registry = registry or ModelRegistry()
    results: dict[str, TrainingResult | str] = {}
    if data["target_regime_label"].nunique(dropna=True) > 1:
        results["supervised_regime"] = train_walk_forward_classifiers(
            data,
            target="target_regime_label",
            model_name_prefix="regime",
            registry=registry,
            train_window=train_window,
            test_window=test_window,
        )
    else:
        results["supervised_regime"] = "Skipped supervised regime: only one regime label"
    results["hmm"] = _train_optional_hmm(data, registry=registry)
    return results


def _train_optional_hmm(data: pd.DataFrame, *, registry: ModelRegistry) -> str:
    try:
        from hmmlearn.hmm import GaussianHMM
    except Exception:
        return "Skipped HMM: hmmlearn package unavailable"
    columns = feature_columns(data, "target_regime_label")[:12]
    if len(data) < 20 or not columns:
        return "Skipped HMM: insufficient observations or features"
    model = GaussianHMM(n_components=3, covariance_type="diag", n_iter=100, random_state=7)
    model.fit(data[columns].fillna(0.0))
    entry = registry.save_model(
        model,
        model_name="hmm_regime",
        model_type="GaussianHMM",
        target="unsupervised_regime_state",
        metadata={"feature_columns": columns},
    )
    return f"Saved HMM: {entry.artifact_path}"

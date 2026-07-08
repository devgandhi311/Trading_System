"""Train breakout-failure/filter signal models."""

from __future__ import annotations

import pandas as pd

from ml.dataset_builder import DatasetBuilder
from ml.model_registry import ModelRegistry
from ml.training_utils import TrainingResult, train_walk_forward_classifiers


def train_signal_model(
    dataset: pd.DataFrame | None = None,
    *,
    registry: ModelRegistry | None = None,
    train_window: int = 252,
    test_window: int = 63,
) -> TrainingResult:
    """Train classifier for breakout-failure/filter target."""

    data = dataset if dataset is not None else DatasetBuilder().build_dataset().dataset
    return train_walk_forward_classifiers(
        data,
        target="target_breakout_failure",
        model_name_prefix="signal_filter",
        registry=registry,
        train_window=train_window,
        test_window=test_window,
    )

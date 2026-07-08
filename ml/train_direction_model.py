"""Train next-day direction models."""

from __future__ import annotations

import pandas as pd

from ml.dataset_builder import DatasetBuilder
from ml.model_registry import ModelRegistry
from ml.training_utils import TrainingResult, train_walk_forward_classifiers


def train_direction_model(
    dataset: pd.DataFrame | None = None,
    *,
    registry: ModelRegistry | None = None,
    train_window: int = 252,
    test_window: int = 63,
) -> TrainingResult:
    """Train Logistic Regression, Random Forest, Gradient Boosting, and optional boosters."""

    data = dataset if dataset is not None else DatasetBuilder().build_dataset().dataset
    return train_walk_forward_classifiers(
        data,
        target="target_next_day_direction",
        model_name_prefix="direction",
        registry=registry,
        train_window=train_window,
        test_window=test_window,
    )

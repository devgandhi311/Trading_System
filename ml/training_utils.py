"""Shared training utilities for walk-forward model validation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error

from ml.dataset_builder import DatasetBuilder, WalkForwardSplit
from ml.model_registry import ModelRegistry, ModelRegistryEntry
from utils.file_utils import ensure_parent_dir

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingResult:
    """Training artifacts and reports."""

    best_model_name: str
    registry_entry: ModelRegistryEntry | None
    metrics_path: Path
    feature_importance_path: Path | None
    metrics: pd.DataFrame


def feature_columns(dataset: pd.DataFrame, target: str) -> list[str]:
    """Return numeric feature columns excluding targets and leakage-prone metadata."""

    excluded_prefixes = ("target_",)
    excluded = {"date", "source_file", target}
    columns = []
    for column in dataset.columns:
        if column in excluded or column.startswith(excluded_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(dataset[column]):
            columns.append(column)
    return columns


def train_walk_forward_classifiers(
    dataset: pd.DataFrame,
    *,
    target: str,
    model_name_prefix: str,
    report_dir: str | Path = "reports/model_reports",
    registry: ModelRegistry | None = None,
    train_window: int = 252,
    test_window: int = 63,
) -> TrainingResult:
    """Train baseline classifiers with walk-forward validation."""

    models: dict[str, Any] = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=7),
        "gradient_boosting": GradientBoostingClassifier(random_state=7),
    }
    _add_optional_boosters(models, task="classification")
    return _train_walk_forward(
        dataset,
        target=target,
        models=models,
        model_name_prefix=model_name_prefix,
        report_dir=report_dir,
        registry=registry,
        train_window=train_window,
        test_window=test_window,
        classification=True,
    )


def train_walk_forward_regressors(
    dataset: pd.DataFrame,
    *,
    target: str,
    model_name_prefix: str,
    report_dir: str | Path = "reports/model_reports",
    registry: ModelRegistry | None = None,
    train_window: int = 252,
    test_window: int = 63,
) -> TrainingResult:
    """Train baseline regressors with walk-forward validation."""

    models: dict[str, Any] = {
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=7),
        "gradient_boosting": GradientBoostingRegressor(random_state=7),
    }
    _add_optional_boosters(models, task="regression")
    return _train_walk_forward(
        dataset,
        target=target,
        models=models,
        model_name_prefix=model_name_prefix,
        report_dir=report_dir,
        registry=registry,
        train_window=train_window,
        test_window=test_window,
        classification=False,
    )


def _train_walk_forward(
    dataset: pd.DataFrame,
    *,
    target: str,
    models: dict[str, Any],
    model_name_prefix: str,
    report_dir: str | Path,
    registry: ModelRegistry | None,
    train_window: int,
    test_window: int,
    classification: bool,
) -> TrainingResult:
    data = dataset.dropna(subset=[target]).reset_index(drop=True)
    columns = feature_columns(data, target)
    splits = DatasetBuilder.walk_forward_splits(data, train_window=train_window, test_window=test_window)
    if not splits:
        fallback_train = max(1, int(len(data) * 0.7))
        splits = (WalkForwardSplit(0, 0, fallback_train, fallback_train, len(data)),)
    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, Any] = {}
    for model_key, model in models.items():
        if classification and model_key == "xgboost" and not pd.api.types.is_numeric_dtype(data[target]):
            LOGGER.info(
                "Skipping optional XGBoost classifier for non-numeric target",
                extra={"target": target},
            )
            continue
        fold_scores: list[float] = []
        for split in splits:
            train = data.iloc[split.train_start : split.train_end]
            test = data.iloc[split.test_start : split.test_end]
            if test.empty or train.empty:
                continue
            x_train = train[columns].fillna(0.0)
            y_train = train[target]
            x_test = test[columns].fillna(0.0)
            y_test = test[target]
            if classification and y_train.nunique() < 2:
                continue
            try:
                model.fit(x_train, y_train)
            except Exception:
                if model_key in {"xgboost", "lightgbm"}:
                    LOGGER.warning(
                        "Skipping optional model fold after fit failure",
                        extra={"model": model_key, "target": target, "fold": split.fold},
                        exc_info=True,
                    )
                    continue
                raise
            predictions = model.predict(x_test)
            if classification:
                score = accuracy_score(y_test, predictions)
                metric_name = "accuracy"
            else:
                score = mean_absolute_error(y_test, predictions)
                metric_name = "mae"
            fold_scores.append(float(score))
            rows.append({"model": model_key, "fold": split.fold, metric_name: score, "test_rows": len(test)})
        if fold_scores:
            fitted_models[model_key] = model
    metrics = pd.DataFrame(rows)
    report_path = ensure_parent_dir(Path(report_dir) / f"{model_name_prefix}_metrics.xlsx")
    with pd.ExcelWriter(report_path) as writer:
        metrics.to_excel(writer, sheet_name="walk_forward", index=False)
        if not metrics.empty:
            metrics.groupby("model").mean(numeric_only=True).reset_index().to_excel(
                writer, sheet_name="summary", index=False
            )
    if not fitted_models:
        return TrainingResult(model_name_prefix, None, report_path, None, metrics)
    best_key = _select_best_model(metrics, classification=classification)
    best_model = fitted_models[best_key]
    importance_path = _save_feature_importance(best_model, columns, Path(report_dir) / f"{model_name_prefix}_importance.xlsx")
    registry = registry or ModelRegistry()
    entry = registry.save_model(
        best_model,
        model_name=f"{model_name_prefix}_{best_key}",
        model_type=best_key,
        target=target,
        metrics_path=report_path,
        feature_importance_path=importance_path,
        metadata={"feature_columns": columns},
    )
    return TrainingResult(best_key, entry, report_path, importance_path, metrics)


def _select_best_model(metrics: pd.DataFrame, *, classification: bool) -> str:
    if metrics.empty:
        raise ValueError("Cannot select best model from empty metrics")
    summary = metrics.groupby("model").mean(numeric_only=True)
    if classification:
        return str(summary["accuracy"].idxmax())
    return str(summary["mae"].idxmin())


def _save_feature_importance(model: Any, columns: list[str], path: Path) -> Path:
    output = ensure_parent_dir(path)
    if hasattr(model, "feature_importances_"):
        values = model.feature_importances_
    elif hasattr(model, "coef_"):
        values = np.ravel(np.abs(model.coef_))
        if values.size != len(columns):
            values = np.resize(values, len(columns))
    else:
        values = np.zeros(len(columns))
    pd.DataFrame({"feature": columns, "importance": values}).sort_values(
        "importance", ascending=False
    ).to_excel(output, index=False)
    return output


def _add_optional_boosters(models: dict[str, Any], *, task: str) -> None:
    try:
        if task == "classification":
            from xgboost import XGBClassifier

            models["xgboost"] = XGBClassifier(eval_metric="logloss", random_state=7)
        else:
            from xgboost import XGBRegressor

            models["xgboost"] = XGBRegressor(random_state=7)
    except Exception:
        pass
    try:
        if task == "classification":
            from lightgbm import LGBMClassifier

            models["lightgbm"] = LGBMClassifier(random_state=7)
        else:
            from lightgbm import LGBMRegressor

            models["lightgbm"] = LGBMRegressor(random_state=7)
    except Exception:
        pass

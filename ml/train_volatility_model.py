"""Train volatility and range models."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any
import warnings

import pandas as pd

from ml.dataset_builder import DatasetBuilder
from ml.model_registry import ModelRegistry
from ml.training_utils import TrainingResult, train_walk_forward_classifiers, train_walk_forward_regressors
from utils.file_utils import ensure_parent_dir


def train_volatility_model(
    dataset: pd.DataFrame | None = None,
    *,
    registry: ModelRegistry | None = None,
    report_dir: str | Path = "reports/model_reports",
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
            report_dir=report_dir,
            registry=registry,
            train_window=train_window,
            test_window=test_window,
        ),
        "expected_range": train_walk_forward_regressors(
            data,
            target="target_expected_range",
            model_name_prefix="expected_range",
            report_dir=report_dir,
            registry=registry,
            train_window=train_window,
            test_window=test_window,
        ),
    }
    results["garch"] = _train_optional_arch(data, registry=registry, model_name="garch", vol="GARCH")
    results["egarch"] = _train_optional_arch(data, registry=registry, model_name="egarch", vol="EGARCH")
    report_path = ensure_parent_dir(Path(report_dir) / "volatility_optional_models.xlsx")
    evidence = pd.DataFrame(
        [
            _optional_model_evidence("garch", "GARCH", str(results["garch"])),
            _optional_model_evidence("egarch", "EGARCH", str(results["egarch"])),
        ],
        columns=("model_name", "model_type", "outcome", "status", "reason", "artifact_path"),
    )
    with pd.ExcelWriter(report_path) as writer:
        evidence.to_excel(writer, sheet_name="optional_models", index=False)
    results["optional_models_report"] = str(report_path)
    return results


def _optional_model_evidence(model_name: str, model_type: str, result: str) -> dict[str, str]:
    """Convert the stable optional-model result contract into qualification evidence."""

    saved_prefix = f"Saved {model_name}: "
    skipped_prefix = f"Skipped {model_name}: "
    if result.startswith(saved_prefix):
        return {
            "model_name": model_name,
            "model_type": model_type,
            "outcome": "saved",
            "status": "qualified",
            "reason": "",
            "artifact_path": result[len(saved_prefix):],
        }

    reason = result[len(skipped_prefix):] if result.startswith(skipped_prefix) else result
    status = "rejected" if "did not converge" in reason.lower() else "not_qualified"
    return {
        "model_name": model_name,
        "model_type": model_type,
        "outcome": "skipped",
        "status": status,
        "reason": reason,
        "artifact_path": "",
    }


def _train_optional_arch(
    data: pd.DataFrame,
    *,
    registry: ModelRegistry,
    model_name: str,
    vol: str,
) -> str:
    try:
        from arch import arch_model
        from arch.utility.exceptions import ConvergenceWarning
    except Exception:
        return f"Skipped {model_name}: arch package unavailable"
    returns = pd.to_numeric(data.get("target_next_day_return"), errors="coerce").dropna() * 100
    if len(returns) < 20:
        return f"Skipped {model_name}: insufficient observations"
    model = arch_model(returns, vol=vol, p=1, q=1, rescale=False)
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always", ConvergenceWarning)
        fitted = model.fit(disp="off", show_warning=True, options={"maxiter": 1_000})

    optimization = getattr(fitted, "optimization_result", None)
    convergence_flag = getattr(fitted, "convergence_flag", None)
    optimizer_success = getattr(optimization, "success", None)
    optimizer_status = getattr(optimization, "status", None)
    convergence_warnings = [item for item in captured if issubclass(item.category, ConvergenceWarning)]
    if convergence_flag != 0 or optimizer_success is not True or optimizer_status != 0 or convergence_warnings:
        message = str(getattr(optimization, "message", "optimizer did not report success"))
        return (
            f"Skipped {model_name}: optimization did not converge "
            f"(flag={convergence_flag}, status={optimizer_status}, message={message})"
        )

    entry = registry.save_model(fitted, model_name=model_name, model_type=vol, target="target_next_day_return")
    return f"Saved {model_name}: {entry.artifact_path}"

"""Optional HMM regime model for quant research."""

from __future__ import annotations

from dataclasses import dataclass, field
import pickle
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class HMMRegimeModel:
    """Gaussian HMM wrapper with a deterministic fallback when hmmlearn is unavailable."""

    n_states: int = 3
    covariance_type: str = "diag"
    random_state: int = 7
    n_iter: int = 100
    model: Any | None = None
    fallback_centers: np.ndarray | None = None
    fitted: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def fit(self, features: np.ndarray | list[list[float]]) -> "HMMRegimeModel":
        """Fit HMM states to feature observations."""

        matrix = _feature_matrix(features)
        if matrix.shape[0] < self.n_states:
            raise ValueError("Not enough observations to fit HMM regime model")
        try:
            from hmmlearn.hmm import GaussianHMM

            self.model = GaussianHMM(
                n_components=self.n_states,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self.model.fit(matrix)
            self.metadata["model_used"] = "GaussianHMM"
        except Exception as exc:
            self.model = None
            self.fallback_centers = _quantile_centers(matrix, self.n_states)
            self.metadata["model_used"] = "quantile_fallback"
            self.metadata["warning"] = str(exc)
        self.fitted = True
        return self

    def predict_state(self, features: np.ndarray | list[list[float]]) -> int:
        """Predict the most likely current hidden regime state."""

        matrix = _feature_matrix(features)
        self._ensure_fitted()
        if self.model is not None:
            return int(self.model.predict(matrix)[-1])
        probabilities = self.predict_proba(matrix)
        return int(np.argmax(probabilities))

    def predict_proba(self, features: np.ndarray | list[list[float]]) -> dict[int, float]:
        """Return hidden-state probabilities for the latest observation."""

        matrix = _feature_matrix(features)
        self._ensure_fitted()
        if self.model is not None:
            probabilities = np.asarray(self.model.predict_proba(matrix)[-1], dtype=float)
        else:
            assert self.fallback_centers is not None
            latest = matrix[-1]
            distances = np.linalg.norm(self.fallback_centers - latest, axis=1)
            inverse = 1.0 / np.maximum(distances, 1e-9)
            probabilities = inverse / inverse.sum()
        return {state: float(probabilities[state]) for state in range(len(probabilities))}

    def save(self, path: str | Path) -> Path:
        """Persist the HMM wrapper to disk."""

        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("wb") as file:
            pickle.dump(self, file)
        return file_path

    @classmethod
    def load(cls, path: str | Path) -> "HMMRegimeModel":
        """Load a persisted HMM wrapper."""

        with Path(path).open("rb") as file:
            value = pickle.load(file)
        if not isinstance(value, cls):
            raise TypeError("Saved object is not an HMMRegimeModel")
        return value

    def _ensure_fitted(self) -> None:
        if not self.fitted:
            raise ValueError("HMMRegimeModel must be fitted before prediction")


def _feature_matrix(features: np.ndarray | list[list[float]]) -> np.ndarray:
    matrix = np.asarray(features, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix.reshape(-1, 1)
    if matrix.ndim != 2:
        raise ValueError("features must be a 1D or 2D numeric array")
    if not np.isfinite(matrix).all():
        raise ValueError("features contain non-finite values")
    return matrix


def _quantile_centers(matrix: np.ndarray, n_states: int) -> np.ndarray:
    order_source = matrix[:, 0]
    quantiles = np.linspace(0.0, 1.0, n_states + 2)[1:-1]
    thresholds = np.quantile(order_source, quantiles)
    labels = np.digitize(order_source, thresholds)
    centers = []
    for state in range(n_states):
        bucket = matrix[labels == state]
        centers.append(bucket.mean(axis=0) if bucket.size else matrix.mean(axis=0))
    return np.asarray(centers, dtype=float)

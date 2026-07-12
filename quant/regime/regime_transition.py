"""Regime transition probability research helpers."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


class RegimeTransitionModel:
    """Estimate transition probabilities from observed regime labels or states."""

    def transition_matrix(self, regimes: Sequence[str | int]) -> dict[str | int, dict[str | int, float]]:
        """Return a normalized transition matrix."""

        labels = list(regimes)
        if len(labels) < 2:
            return {}
        states = sorted(set(labels), key=str)
        counts = {state: {next_state: 0.0 for next_state in states} for state in states}
        for current, next_state in zip(labels[:-1], labels[1:]):
            counts[current][next_state] += 1.0
        matrix: dict[str | int, dict[str | int, float]] = {}
        for state, row in counts.items():
            total = sum(row.values())
            if total <= 0:
                matrix[state] = {next_state: 1.0 / len(states) for next_state in states}
            else:
                matrix[state] = {next_state: value / total for next_state, value in row.items()}
        return matrix

    def probability_next_regime(
        self,
        current_regime: str | int,
        regimes: Sequence[str | int] | None = None,
        matrix: dict[str | int, dict[str | int, float]] | None = None,
    ) -> dict[str | int, float]:
        """Return next-regime probability row for current regime."""

        transition = matrix or self.transition_matrix(regimes or ())
        if current_regime in transition:
            return dict(transition[current_regime])
        if not transition:
            return {}
        states = list(transition)
        return {state: 1.0 / len(states) for state in states}


def transition_risk(next_regime_probability: dict[str | int, float], current_regime: str | int) -> float:
    """Estimate risk that the current regime changes on the next step."""

    if not next_regime_probability:
        return 0.0
    stay_probability = float(next_regime_probability.get(current_regime, 0.0))
    return float(np.clip(1.0 - stay_probability, 0.0, 1.0))

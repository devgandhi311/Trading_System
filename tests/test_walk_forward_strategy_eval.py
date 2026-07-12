"""Tests for walk-forward strategy evaluation."""

from __future__ import annotations

import pandas as pd

from quant.optimization.walk_forward_strategy_eval import evaluate_walk_forward, save_walk_forward_report


def test_walk_forward_strategy_eval_saves_report(tmp_path) -> None:
    data = pd.DataFrame(
        {
            "date": pd.date_range("2024-01-01", periods=180, freq="B"),
            "close": [26000.0 + index * 2.0 for index in range(180)],
        }
    )

    rows = evaluate_walk_forward(data, train_size=60, test_size=20)
    path = save_walk_forward_report(rows, tmp_path / "walk_forward.xlsx")

    assert rows
    assert path.exists()
    assert {row.strategy for row in rows} == {
        "ExpectedMoveCreditSpread",
        "ExpectedMoveIronCondor",
        "IVMeanReversionPremium",
        "VolatilityExpansion",
    }

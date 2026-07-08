"""Walk-forward validation utilities for backtesting."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class WalkForwardWindow:
    """One walk-forward train/test window."""

    fold: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int

    def train(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return train slice."""

        return dataframe.iloc[self.train_start:self.train_end].copy()

    def test(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Return test slice."""

        return dataframe.iloc[self.test_start:self.test_end].copy()


class WalkForwardSplitter:
    """Create expanding-window walk-forward splits."""

    def __init__(self, *, train_size: int = 252, test_size: int = 63, step_size: int | None = None) -> None:
        """Create splitter."""

        self.train_size = train_size
        self.test_size = test_size
        self.step_size = step_size or test_size

    def split(self, dataframe: pd.DataFrame) -> tuple[WalkForwardWindow, ...]:
        """Return walk-forward windows for a dataframe."""

        windows: list[WalkForwardWindow] = []
        fold = 0
        train_start = 0
        train_end = min(self.train_size, len(dataframe))
        while train_end < len(dataframe):
            test_start = train_end
            test_end = min(test_start + self.test_size, len(dataframe))
            if test_end <= test_start:
                break
            windows.append(WalkForwardWindow(fold, train_start, train_end, test_start, test_end))
            fold += 1
            train_end = min(train_end + self.step_size, len(dataframe))
        return tuple(windows)


def run_walk_forward_validation(
    dataframe: pd.DataFrame,
    runner,
    *,
    splitter: WalkForwardSplitter | None = None,
) -> pd.DataFrame:
    """Run a callable over walk-forward test folds and collect summaries."""

    active_splitter = splitter or WalkForwardSplitter()
    rows = []
    for window in active_splitter.split(dataframe):
        result = runner(window.train(dataframe), window.test(dataframe), window)
        row = {"fold": window.fold, "train_rows": window.train_end - window.train_start, "test_rows": window.test_end - window.test_start}
        if isinstance(result, dict):
            row.update(result)
        rows.append(row)
    return pd.DataFrame(rows)

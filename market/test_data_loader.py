"""Manual historical data loader diagnostic.

Run:
    python market/test_data_loader.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from market.data_loader import DataLoader
from utils.exceptions import DataValidationError


def main() -> int:
    """Print real NSE data discovery and loading statistics."""

    loader = DataLoader()
    files = loader._discover_files()
    print(f"Files loaded: {len(files)}")
    for path in files:
        print(f"- {path.name}")
    if not files:
        print("Rows loaded: 0")
        print("First date: N/A")
        print("Last date: N/A")
        print("Detected OHLC columns: N/A")
        print("Missing rows: N/A")
        return 1
    try:
        frames = [loader._read_file(path) for path in files]
        raw = pd.concat(frames, ignore_index=True)
        clean, stats = loader.clean(raw)
    except DataValidationError as exc:
        print(f"Loader error: {exc}")
        return 1

    print(f"Rows loaded: {len(clean)}")
    print(f"First date: {clean['date'].min().date() if not clean.empty else 'N/A'}")
    print(f"Last date: {clean['date'].max().date() if not clean.empty else 'N/A'}")
    print("Detected OHLC columns: date, open, high, low, close, volume")
    print(f"Missing rows: {stats['removed_missing_rows']}")
    print(f"Bad OHLC rows: {stats['removed_bad_ohlc_rows']}")
    print(f"Duplicate rows: {stats['removed_duplicate_rows']}")
    print(f"Columns: {', '.join(clean.columns)}")
    print(f"Missing values: {int(clean.isna().sum().sum())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

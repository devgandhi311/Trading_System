"""Inspect real NSE raw data files without changing processing state.

Run:
    python scripts/check_data_files.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market.data_loader import DataLoader
from utils.config_loader import ConfigLoader


SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".parquet"}


def main() -> int:
    """Print per-file diagnostics for raw NSE data files."""

    loader = DataLoader(config_loader=ConfigLoader())
    data_dir = loader.raw_dir.resolve()
    files = _discover_files(data_dir)

    print(f"project root: {PROJECT_ROOT}")
    print(f"data folder path: {data_dir}")
    print(f"number of files found: {len(files)}")
    if files:
        print("file names:")
        for path in files:
            print(f"- {path.name}")
        print(f"file extensions: {', '.join(sorted({path.suffix.lower() for path in files}))}")
    else:
        print("file names: none")
        print("file extensions: none")
        return 0

    for path in files:
        print("")
        print(f"file: {path.name}")
        print(f"extension: {path.suffix.lower()}")
        try:
            frame = _read_file(path)
        except Exception as exc:
            print(f"error: {type(exc).__name__}: {exc}")
            continue

        columns = [str(column) for column in frame.columns]
        detection = _detect_columns(frame, loader)
        print(f"rows per file: {len(frame)}")
        print(f"detected columns: {', '.join(columns)}")
        print(f"detected date column: {detection.get('date', 'NOT FOUND')}")
        print(
            "detected OHLCV columns: "
            f"open={detection.get('open', 'NOT FOUND')}, "
            f"high={detection.get('high', 'NOT FOUND')}, "
            f"low={detection.get('low', 'NOT FOUND')}, "
            f"close={detection.get('close', 'NOT FOUND')}, "
            f"volume={detection.get('volume', 'NOT FOUND')}"
        )
        date_column = detection.get("date")
        if date_column:
            dates = pd.to_datetime(frame[date_column], errors="coerce").dropna()
            first_date = dates.min().date().isoformat() if not dates.empty else "N/A"
            last_date = dates.max().date().isoformat() if not dates.empty else "N/A"
        else:
            first_date = "N/A"
            last_date = "N/A"
        print(f"first date: {first_date}")
        print(f"last date: {last_date}")
        print(f"missing values: {int(frame.isna().sum().sum())}")
    return 0


def _discover_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(path for path in data_dir.rglob("*") if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES)


def _read_file(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {suffix}")


def _detect_columns(frame: pd.DataFrame, loader: DataLoader) -> dict[str, str]:
    normalized_columns = {loader._normalize_column_name(column): str(column) for column in frame.columns}
    detected: dict[str, str] = {}
    for canonical, aliases in loader.COLUMN_ALIASES.items():
        match = next((normalized_columns[alias] for alias in aliases if alias in normalized_columns), None)
        if match:
            detected[canonical] = match
    return detected


if __name__ == "__main__":
    raise SystemExit(main())

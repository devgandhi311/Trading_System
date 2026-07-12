"""Run the actual DataLoader and print a clean NSE data summary.

Run:
    python scripts/check_data_loader.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from market.data_loader import DataLoader
from utils.config_loader import ConfigLoader


def main() -> int:
    """Load real NSE files through DataLoader and print summary."""

    loader = DataLoader(config_loader=ConfigLoader())
    data, report = loader.load_30_years_nse_data(dataset_name="nse_30_years_clean")

    print(f"rows loaded: {len(data)}")
    print(f"first date: {data['date'].min().date().isoformat()}")
    print(f"last date: {data['date'].max().date().isoformat()}")
    print(f"columns: {', '.join(str(column) for column in data.columns)}")
    print(f"bad rows removed: {report.removed_bad_ohlc_rows}")
    print(f"missing rows removed: {report.removed_missing_rows}")
    print(f"duplicate rows removed: {report.removed_duplicate_rows}")
    print(f"processed file path: {report.output_path}")
    print(f"quality report path: {report.report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

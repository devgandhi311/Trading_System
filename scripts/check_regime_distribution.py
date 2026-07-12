"""Diagnose historical regime distribution without changing regime logic.

Run:
    python scripts/check_regime_distribution.py
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from features.microstructure import add_microstructure_features
from features.returns import add_return_features
from features.volatility import add_volatility_features
from market.data_loader import DataLoader
from regime.regime_detector import RegimeDetector
from utils.config_loader import ConfigLoader
from utils.file_utils import ensure_parent_dir


OUTPUT_PATH = PROJECT_ROOT / "reports" / "risk_reports" / "regime_distribution.csv"
LOOKBACK_BARS = 60


def main() -> int:
    """Load real data, generate features, classify regimes, and save counts."""

    config_loader = ConfigLoader()
    data_loader = DataLoader(config_loader=config_loader)
    historical_data, _report = data_loader.load_30_years_nse_data(dataset_name="nse_30_years_clean")

    features = _generate_detector_features(historical_data)
    detector = RegimeDetector()
    regime_names: list[str] = []
    for index in range(len(features)):
        start = max(0, index - LOOKBACK_BARS + 1)
        result = detector.detect(features.iloc[start : index + 1], symbol="NIFTY")
        regime_names.append(result.regime_name or result.regime.value)

    total_rows = len(features)
    counts = pd.Series(regime_names, name="regime").value_counts().rename_axis("regime").reset_index(name="count")
    counts["percentage"] = counts["count"] / max(total_rows, 1) * 100.0
    output_path = ensure_parent_dir(OUTPUT_PATH)
    counts.to_csv(output_path, index=False)

    dates = pd.to_datetime(features["date"], errors="coerce").dropna()
    print(f"total rows: {total_rows}")
    print("regime counts:")
    for row in counts.itertuples(index=False):
        print(f"- {row.regime}: {row.count}")
    print("regime percentages:")
    for row in counts.itertuples(index=False):
        print(f"- {row.regime}: {row.percentage:.2f}%")
    print(f"first date: {dates.min().date().isoformat() if not dates.empty else 'N/A'}")
    print(f"last date: {dates.max().date().isoformat() if not dates.empty else 'N/A'}")
    print(f"saved: {output_path}")
    return 0


def _generate_detector_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Generate only features consumed by the current regime detector."""

    features = add_return_features(dataframe)
    features = add_volatility_features(features)
    features = add_microstructure_features(features)
    features["smart_money_context_strength"] = 0.0
    features["failed_breakout_detected"] = False
    features["failed_breakout_strength"] = 0.0
    features["liquidity_sweep_above_previous_high_detected"] = False
    features["liquidity_sweep_below_previous_low_detected"] = False
    features["liquidity_sweep_above_previous_high_strength"] = 0.0
    features["liquidity_sweep_below_previous_low_strength"] = 0.0
    return features


if __name__ == "__main__":
    raise SystemExit(main())

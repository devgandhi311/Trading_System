"""Unified feature pipeline for regime detection, prediction, and filtering."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from features.microstructure import add_microstructure_features
from features.regime_features import create_regime_feature_table
from features.returns import add_return_features
from features.smart_money import (
    bearish_fair_value_gap,
    bearish_order_block,
    break_of_structure,
    bullish_fair_value_gap,
    bullish_order_block,
    change_of_character,
    displacement_candle,
    equal_highs,
    equal_lows,
    failed_breakout,
    liquidity_sweep_above_previous_high,
    liquidity_sweep_below_previous_low,
)
from features.volatility import add_volatility_features
from market.data_store import DataStore
from models import Mode
from utils.exceptions import DataValidationError
from utils.file_utils import ensure_parent_dir


@dataclass(frozen=True)
class FeaturePipelineResult:
    """Saved feature outputs and reports."""

    features: pd.DataFrame
    regime_features: pd.DataFrame
    features_path: Path
    regime_features_path: Path
    summary_report_path: Path


class FeaturePipeline:
    """Build and cache statistical, volatility, microstructure, and smart-money features."""

    REQUIRED_COLUMNS = {"date", "open", "high", "low", "close", "volume"}

    def __init__(
        self,
        *,
        mode: Mode | str = Mode.PAPER,
        data_store: DataStore | None = None,
        summary_report_path: str | Path = "reports/risk_reports/feature_summary_report.xlsx",
    ) -> None:
        """Create a feature pipeline for BACKTEST, PAPER, or LIVE mode."""

        self.mode = Mode(str(mode).upper())
        self.data_store = data_store or DataStore()
        self.summary_report_path = Path(summary_report_path)

    def generate_features(
        self,
        dataframe: pd.DataFrame,
        *,
        dataset_name: str = "combined_features",
        rolling_window: int = 20,
        cache: bool = True,
        latest_news_sentiment_score: float = 0.0,
        latest_event_risk_score: float = 0.0,
    ) -> FeaturePipelineResult:
        """Generate combined features, regime table, cache outputs, and write a summary report."""

        data = self._validate_and_prepare(dataframe)
        features = add_return_features(data, rolling_window=rolling_window, probability_window=rolling_window)
        features = add_volatility_features(features, rolling_window=rolling_window)
        features = add_microstructure_features(features, rolling_window=rolling_window)
        features = self._add_smart_money_features(features)
        features["news_sentiment_score"] = float(latest_news_sentiment_score)
        features["event_risk_score"] = float(latest_event_risk_score)
        features["feature_mode"] = self.mode.value
        regime_features = create_regime_feature_table(features)

        if cache:
            features_path = self.data_store.save_features(features, dataset_name)
            regime_path = self.data_store.save_features(regime_features, f"{dataset_name}_regime")
        else:
            features_path = self.data_store.features_dir / f"{dataset_name}.csv"
            regime_path = self.data_store.features_dir / f"{dataset_name}_regime.csv"
        summary_path = self.generate_feature_summary_report(features, regime_features)
        return FeaturePipelineResult(
            features=features,
            regime_features=regime_features,
            features_path=features_path,
            regime_features_path=regime_path,
            summary_report_path=summary_path,
        )

    def generate_regime_feature_table(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Generate only the regime-specific feature table."""

        data = self._validate_and_prepare(dataframe)
        features = add_return_features(data)
        features = add_volatility_features(features)
        features = add_microstructure_features(features)
        features = self._add_smart_money_features(features)
        return create_regime_feature_table(features)

    def generate_feature_summary_report(
        self,
        features: pd.DataFrame,
        regime_features: pd.DataFrame,
        path: str | Path | None = None,
    ) -> Path:
        """Write an Excel summary report for generated features."""

        report_path = ensure_parent_dir(path or self.summary_report_path)
        summary_rows = []
        for column in features.columns:
            series = features[column]
            summary_rows.append(
                {
                    "feature": column,
                    "dtype": str(series.dtype),
                    "missing_count": int(series.isna().sum()),
                    "missing_percent": float(series.isna().mean()),
                    "unique_count": int(series.nunique(dropna=True)),
                    "is_regime_feature": column in regime_features.columns,
                }
            )
        with pd.ExcelWriter(report_path) as writer:
            pd.DataFrame(summary_rows).to_excel(writer, sheet_name="summary", index=False)
            pd.DataFrame(
                [
                    {"metric": "mode", "value": self.mode.value},
                    {"metric": "feature_rows", "value": len(features)},
                    {"metric": "feature_columns", "value": len(features.columns)},
                    {"metric": "regime_feature_columns", "value": len(regime_features.columns)},
                ]
            ).to_excel(writer, sheet_name="metadata", index=False)
        return report_path

    def _validate_and_prepare(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        missing = self.REQUIRED_COLUMNS.difference(dataframe.columns)
        if missing:
            raise DataValidationError(f"Missing required feature columns: {sorted(missing)}")
        data = dataframe.copy()
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data = data.dropna(subset=["date"]).sort_values("date").drop_duplicates(subset=["date"], keep="last")
        for column in ("open", "high", "low", "close", "volume"):
            data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=["open", "high", "low", "close", "volume"])
        if data.empty:
            raise DataValidationError("No valid rows available for feature pipeline")
        return data.reset_index(drop=True)

    def _add_smart_money_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        data = dataframe.copy()
        detectors = {
            "bullish_fvg": bullish_fair_value_gap,
            "bearish_fvg": bearish_fair_value_gap,
            "liquidity_sweep_above_previous_high": liquidity_sweep_above_previous_high,
            "liquidity_sweep_below_previous_low": liquidity_sweep_below_previous_low,
            "bullish_order_block": bullish_order_block,
            "bearish_order_block": bearish_order_block,
            "break_of_structure": break_of_structure,
            "change_of_character": change_of_character,
            "equal_highs": equal_highs,
            "equal_lows": equal_lows,
            "failed_breakout": failed_breakout,
            "displacement_candle": displacement_candle,
        }
        for name, detector in detectors.items():
            detected_values: list[bool] = []
            strength_values: list[float] = []
            for index in range(len(data)):
                window = data.iloc[: index + 1]
                result = detector(window)
                detected_values.append(result.detected)
                strength_values.append(result.strength)
            data[f"{name}_detected"] = detected_values
            data[f"{name}_strength"] = strength_values
        data["smart_money_context_strength"] = data[
            [column for column in data.columns if column.endswith("_strength")]
        ].mean(axis=1)
        return data

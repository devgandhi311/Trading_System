"""Latency tracking for market-data-to-order decision flow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import logging
from pathlib import Path
from statistics import median
from time import perf_counter
from typing import Any

from models import LatencyReport, Mode, StrategyType
from utils.config_loader import ConfigLoader
from utils.file_utils import save_excel
from utils.math_utils import clamp
from utils.time_utils import now_ist

LOGGER = logging.getLogger(__name__)

LATENCY_STAGES: tuple[str, ...] = (
    "live_tick_received",
    "feature_update",
    "regime_detection",
    "ml_prediction",
    "strategy_selection",
    "risk_validation",
    "margin_validation",
    "order_api_request",
    "exchange_acknowledgement",
)


@dataclass(frozen=True)
class LatencyThresholds:
    """Latency classification thresholds in milliseconds."""

    paper_backtest_warning_ms: float = 500.0
    live_optimized_warning_ms: float = 2_000.0
    stable_warning_ms: float = 5_000.0
    bad_connection_warning_ms: float = 15_000.0
    scalping_reject_ms: float = 2_000.0
    confidence_reduce_ms: float = 500.0
    confidence_reduction_per_second: float = 0.05
    max_confidence_reduction: float = 0.30


@dataclass(frozen=True)
class LatencySample:
    """One measured latency sample."""

    timestamp: datetime
    component: str
    latency_ms: float
    strategy_type: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LatencyDecision:
    """Latency policy output for trading decisions."""

    allowed: bool
    adjusted_confidence: float
    warning: str | None
    latency_ms: float
    classification: str
    reject_scalping: bool


class LatencyMonitor:
    """Track latency checkpoints and produce daily risk reports."""

    def __init__(
        self,
        *,
        config_loader: ConfigLoader | None = None,
        thresholds: LatencyThresholds | None = None,
        report_dir: str | Path | None = None,
    ) -> None:
        """Create latency monitor."""

        self.config_loader = config_loader or ConfigLoader()
        self.config = self.config_loader.load()
        self.thresholds = thresholds or LatencyThresholds()
        self.report_dir = Path(report_dir or "reports/risk_reports")
        self.samples: list[LatencySample] = []
        self._marks: dict[str, float] = {}

    def mark(self, component: str, *, metadata: dict[str, Any] | None = None) -> None:
        """Record a named checkpoint timestamp."""

        if component not in LATENCY_STAGES and component != "decision_start":
            LOGGER.debug("Recording custom latency component", extra={"component": component})
        self._marks[component] = perf_counter()
        if metadata:
            self.record(component, 0.0, metadata=metadata | {"checkpoint": True})

    def record(
        self,
        component: str,
        latency_ms: float,
        *,
        strategy_type: StrategyType | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LatencySample:
        """Record an explicit latency sample."""

        sample = LatencySample(
            timestamp=now_ist(),
            component=component,
            latency_ms=max(0.0, float(latency_ms)),
            strategy_type=str(strategy_type.value if isinstance(strategy_type, StrategyType) else strategy_type)
            if strategy_type is not None
            else None,
            metadata=metadata or {},
        )
        self.samples.append(sample)
        warning = self.warning_for_latency(sample.latency_ms)
        if warning:
            LOGGER.warning("Latency threshold exceeded", extra={"component": component, "latency_ms": sample.latency_ms, "warning": warning})
        return sample

    def record_stage_duration(
        self,
        start_component: str,
        end_component: str,
        *,
        component: str | None = None,
        strategy_type: StrategyType | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LatencySample:
        """Record duration between two marked checkpoints."""

        if start_component not in self._marks:
            raise ValueError(f"Missing start latency mark: {start_component}")
        if end_component not in self._marks:
            raise ValueError(f"Missing end latency mark: {end_component}")
        latency_ms = (self._marks[end_component] - self._marks[start_component]) * 1_000.0
        return self.record(
            component or f"{start_component}_to_{end_component}",
            latency_ms,
            strategy_type=strategy_type,
            metadata=metadata,
        )

    def record_total_decision_to_order(
        self,
        *,
        strategy_type: StrategyType | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> LatencySample:
        """Record total decision-to-order latency from decision_start to exchange acknowledgement."""

        return self.record_stage_duration(
            "decision_start",
            "exchange_acknowledgement",
            component="total_decision_to_order_time",
            strategy_type=strategy_type,
            metadata=metadata,
        )

    def assess_trade_latency(
        self,
        *,
        strategy_type: StrategyType | str,
        confidence_score: float,
        latency_ms: float | None = None,
    ) -> LatencyDecision:
        """Apply latency policy: reject scalping, reduce confidence, and warn."""

        measured = float(latency_ms if latency_ms is not None else self.latest_latency_ms())
        classification = self.classify_latency(measured)
        normalized_strategy = str(strategy_type.value if isinstance(strategy_type, StrategyType) else strategy_type)
        reject_scalping = normalized_strategy == StrategyType.VOLATILITY_SCALPING.value and measured > self.thresholds.scalping_reject_ms
        reduction = 0.0
        if measured > self.thresholds.confidence_reduce_ms:
            reduction = min(
                self.thresholds.max_confidence_reduction,
                (measured - self.thresholds.confidence_reduce_ms) / 1_000.0 * self.thresholds.confidence_reduction_per_second,
            )
        adjusted_confidence = clamp(confidence_score - reduction, 0.0, 1.0)
        warning = self.warning_for_latency(measured)
        if warning:
            LOGGER.warning(
                "Latency policy warning",
                extra={
                    "strategy_type": normalized_strategy,
                    "latency_ms": measured,
                    "classification": classification,
                    "adjusted_confidence": adjusted_confidence,
                },
            )
        return LatencyDecision(
            allowed=not reject_scalping,
            adjusted_confidence=adjusted_confidence,
            warning=warning,
            latency_ms=measured,
            classification=classification,
            reject_scalping=reject_scalping,
        )

    def latest_latency_ms(self) -> float:
        """Return latest total latency sample or latest sample if total is unavailable."""

        for sample in reversed(self.samples):
            if sample.component == "total_decision_to_order_time":
                return sample.latency_ms
        return self.samples[-1].latency_ms if self.samples else 0.0

    def classify_latency(self, latency_ms: float) -> str:
        """Classify latency using expected operating ranges."""

        value = float(latency_ms)
        if value < 50.0:
            return "paper_backtest_fast"
        if value <= self.thresholds.paper_backtest_warning_ms:
            return "paper_backtest_expected"
        if value <= self.thresholds.live_optimized_warning_ms:
            return "live_optimized_vps_static_ip"
        if value <= self.thresholds.stable_warning_ms:
            return "normal_stable_setup"
        if value <= self.thresholds.bad_connection_warning_ms:
            return "bad_internet_or_api_delay"
        return "severe_latency"

    def warning_for_latency(self, latency_ms: float) -> str | None:
        """Return warning text when latency exceeds the mode-aware threshold."""

        mode = self.config.settings.mode
        threshold = self.thresholds.paper_backtest_warning_ms if mode in {Mode.PAPER, Mode.BACKTEST} else self.thresholds.stable_warning_ms
        if latency_ms <= threshold:
            return None
        return f"Latency {latency_ms:.1f} ms exceeds {threshold:.1f} ms threshold for {mode.value} mode."

    def summary_reports(self, *, report_date: date | None = None) -> tuple[LatencyReport, ...]:
        """Return component-level latency summaries."""

        target_date = report_date or now_ist().date()
        reports: list[LatencyReport] = []
        for component in sorted({sample.component for sample in self.samples}):
            values = [sample.latency_ms for sample in self.samples if sample.component == component and sample.timestamp.date() == target_date]
            if not values:
                continue
            reports.append(
                LatencyReport(
                    report_date=target_date,
                    generated_at=now_ist(),
                    component=component,
                    samples=len(values),
                    p50_ms=_percentile(values, 50),
                    p95_ms=_percentile(values, 95),
                    p99_ms=_percentile(values, 99),
                    max_ms=max(values),
                    metadata={"classification": self.classify_latency(max(values))},
                )
            )
        return tuple(reports)

    def save_daily_report(self, *, report_date: date | None = None) -> Path:
        """Save daily latency report workbook under reports/risk_reports."""

        target_date = report_date or now_ist().date()
        path = self.report_dir / f"latency_report_{target_date.isoformat()}.xlsx"
        rows: list[dict[str, Any]] = []
        for report in self.summary_reports(report_date=target_date):
            row = asdict(report)
            row["generated_at"] = report.generated_at.isoformat()
            row["report_date"] = report.report_date.isoformat()
            row["metadata"] = str(report.metadata)
            rows.append(row)
        for sample in self.samples:
            if sample.timestamp.date() != target_date:
                continue
            rows.append(
                {
                    "report_date": target_date.isoformat(),
                    "generated_at": now_ist().isoformat(),
                    "component": f"sample:{sample.component}",
                    "samples": 1,
                    "p50_ms": sample.latency_ms,
                    "p95_ms": sample.latency_ms,
                    "p99_ms": sample.latency_ms,
                    "max_ms": sample.latency_ms,
                    "metadata": str({"strategy_type": sample.strategy_type, **sample.metadata}),
                }
            )
        if not rows:
            rows.append(
                {
                    "report_date": target_date.isoformat(),
                    "generated_at": now_ist().isoformat(),
                    "component": "no_samples",
                    "samples": 0,
                    "p50_ms": 0.0,
                    "p95_ms": 0.0,
                    "p99_ms": 0.0,
                    "max_ms": 0.0,
                    "metadata": "{}",
                }
            )
        return save_excel(rows, path, sheet_name="Latency")


def _percentile(values: list[float], percentile: int) -> float:
    """Return nearest-rank percentile."""

    if not values:
        return 0.0
    if percentile == 50:
        return float(median(values))
    sorted_values = sorted(values)
    index = round((percentile / 100.0) * (len(sorted_values) - 1))
    return float(sorted_values[index])

"""Idempotent end-of-day controller for PAPER sessions."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
import shutil
from typing import Any, Callable, Mapping

from reports.daily_report import DailyReportBuilder
from runtime.paper_portfolio_service import PaperPortfolioService
from runtime.position_monitor import PositionMonitor
from storage.repositories import OperationalRepository
from utils.file_utils import ensure_parent_dir
from utils.time_utils import ensure_ist, now_ist


@dataclass(frozen=True)
class EndOfDayResult:
    session_id: str
    status: str
    positions_remaining: int
    orders_remaining: int
    report_path: str | None = None
    dashboard_path: str | None = None
    archives: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


class EndOfDayController:
    """Close, reconcile, snapshot, report and archive one session exactly once."""

    def __init__(self, repository: OperationalRepository, portfolio_service: PaperPortfolioService,
                 position_monitor: PositionMonitor, *, session_id: str,
                 reconciliation_runner: Callable[[], Any], cancel_order: Callable[[str], Any],
                 report_dir: str | Path = "reports/daily", dashboard_dir: str | Path = "data/live",
                 archive_dir: str | Path = "reports/archives", logs_dir: str | Path = "logs",
                 market_data_dir: str | Path = "data/market_cache",
                 clock: Callable[[], datetime] = now_ist) -> None:
        self.repository = repository
        self.portfolio_service = portfolio_service
        self.position_monitor = position_monitor
        self.session_id = session_id
        self.reconciliation_runner = reconciliation_runner
        self.cancel_order = cancel_order
        self.report_dir = Path(report_dir)
        self.dashboard_dir = Path(dashboard_dir)
        self.archive_dir = Path(archive_dir)
        self.logs_dir = Path(logs_dir)
        self.market_data_dir = Path(market_data_dir)
        self.clock = clock

    def run(self) -> EndOfDayResult:
        session = self.repository.get_trading_session(self.session_id)
        if session is None:
            raise ValueError(f"Session {self.session_id} not found")
        prior = dict(session.metadata_json).get("eod") or {}
        if prior.get("status") == "COMPLETED":
            return EndOfDayResult(**prior["result"])
        self._checkpoint("STARTED")
        errors: list[str] = []

        for order in self.repository.session_orders(self.session_id):
            if order.local_state in {"OPEN", "SUBMITTED", "PARTIALLY_FILLED", "TRIGGER_PENDING"}:
                try:
                    self.cancel_order(order.idempotency_key)
                except Exception as exc:
                    errors.append(f"cancel {order.idempotency_key}: {exc}")
        self._checkpoint("ORDERS_CANCELLED")

        for position in self.repository.open_positions():
            if position.session_id != self.session_id:
                continue
            self.repository.update_position(
                position.id, expected_version=position.version,
                metadata_json={**position.metadata_json, "manual_exit": True, "eod_exit": True},
            )
        monitor_result = self.position_monitor.monitor_once()
        errors.extend(monitor_result.blocked)
        self._checkpoint("POSITIONS_PROCESSED")

        try:
            reconciliation = self.reconciliation_runner()
            if getattr(reconciliation, "entry_blocked", False):
                errors.append(f"reconciliation: {getattr(reconciliation, 'outcome', 'blocked')}")
        except Exception as exc:
            errors.append(f"reconciliation: {exc}")
        self._checkpoint("RECONCILED")

        portfolio = self.portfolio_service.initialize()
        portfolio_issues = list(portfolio.get("_reconciliation_issues", ()))
        errors.extend(f"portfolio: {issue}" for issue in portfolio_issues)
        open_positions = [p for p in self.repository.open_positions() if p.session_id == self.session_id]
        active_orders = [o for o in self.repository.session_orders(self.session_id)
                         if o.local_state not in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}]
        report_path = self._save_report(portfolio, errors)
        dashboard_path = self._save_dashboard(portfolio, open_positions, active_orders, errors)
        archives = self._archive_inputs()
        status = "COMPLETED" if not open_positions and not active_orders and not errors else "BLOCKED"
        result = EndOfDayResult(self.session_id, status, len(open_positions), len(active_orders),
                                str(report_path), str(dashboard_path), tuple(str(p) for p in archives), tuple(errors))
        self.repository.update_session_metadata(
            self.session_id,
            eod={"status": status, "completed_at": ensure_ist(self.clock()).isoformat(),
                 "result": asdict(result)},
            final_equity=float(portfolio.get("current_equity", 0.0)),
        )
        if status == "COMPLETED":
            current = self.repository.get_trading_session(self.session_id)
            if current and current.status != "CLOSED":
                self.repository.update_session_status(self.session_id, "CLOSED", previous_state=current.status,
                                                      reason="End-of-day completed")
        return result

    def _checkpoint(self, status: str) -> None:
        self.repository.update_session_metadata(
            self.session_id, eod={"status": status, "updated_at": ensure_ist(self.clock()).isoformat()}
        )

    def _save_report(self, portfolio: Mapping[str, Any], errors: list[str]) -> Path:
        exits = self.repository.exit_decisions(self.session_id)
        trades = [{"trade_id": row.position_id, "net_pnl": row.metadata_json.get("realized_pnl", 0.0),
                   "strategy_type": row.metadata_json.get("strategy_id", "unknown")}
                  for row in exits if row.metadata_json.get("status") == "COMPLETED"]
        summary = DailyReportBuilder(
            capital=float(portfolio.get("initial_capital", 0.0)), max_daily_loss_percent=2.0
        ).build(trades, report_date=ensure_ist(self.clock()).date())
        payload = {"session_id": self.session_id, "portfolio": dict(portfolio),
                   "performance": summary.as_dict(), "errors": errors}
        path = ensure_parent_dir(self.report_dir / f"{self.session_id}.json")
        self._atomic_json(path, payload)
        return path

    def _save_dashboard(self, portfolio: Mapping[str, Any], positions: list[Any],
                        orders: list[Any], errors: list[str]) -> Path:
        payload = {"session_id": self.session_id, "status": "BLOCKED" if errors else "COMPLETE",
                   "portfolio": dict(portfolio), "open_positions": [p.id for p in positions],
                   "active_orders": [o.idempotency_key for o in orders], "errors": errors,
                   "updated_at": ensure_ist(self.clock()).isoformat()}
        path = ensure_parent_dir(self.dashboard_dir / "end_of_day.json")
        self._atomic_json(path, payload)
        return path

    def _archive_inputs(self) -> list[Path]:
        outputs: list[Path] = []
        self.archive_dir.mkdir(parents=True, exist_ok=True)
        for label, source in (("logs", self.logs_dir), ("market_data", self.market_data_dir)):
            destination = self.archive_dir / f"{self.session_id}-{label}"
            zip_path = Path(str(destination) + ".zip")
            if zip_path.exists() or not source.exists():
                if zip_path.exists():
                    outputs.append(zip_path)
                continue
            outputs.append(Path(shutil.make_archive(str(destination), "zip", root_dir=source)))
        return outputs

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
        temporary.replace(path)

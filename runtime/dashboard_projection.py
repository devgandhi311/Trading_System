"""Crash-safe dashboard projections rebuilt exclusively from SQLite state."""

from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping

from storage.repositories import OperationalRepository
from utils.time_utils import ensure_ist, now_ist


class AuthoritativeDashboardProjector:
    """Project authoritative operational rows; never invent dashboard values."""

    def __init__(self, repository: OperationalRepository, state_dir: str | Path,
                 *, clock=now_ist) -> None:
        self.repository = repository
        self.state_dir = Path(state_dir)
        self.clock = clock

    def project(self, session_id: str, *, action: str) -> dict[str, Any]:
        session = self.repository.get_trading_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id} not found")
        decision = self.repository.latest_decision_context_for_session(session_id)
        portfolio = self.repository.latest_portfolio_for_session(session_id)
        reconciliation = self.repository.latest_reconciliation(session_id)
        kill = self.repository.current_kill_switch()
        orders = self.repository.session_orders(session_id)
        fills = self.repository.session_fills(session_id)
        positions = [row for row in self.repository.open_positions() if row.session_id == session_id]
        exits = self.repository.exit_decisions(session_id)
        generated_at = ensure_ist(self.clock()).isoformat()

        unavailable = {"available": False, "reason": "not persisted in authoritative database"}
        prediction = unavailable if decision is None else {
            "available": True, "decision_cycle_id": decision.decision_cycle_id,
            "timestamp": decision.timestamp, "probabilities": decision.prediction_probabilities,
            "confidence": decision.prediction_confidence, "expected_return_percent": decision.expected_return_percent,
            "expected_range_low": decision.expected_range_low, "expected_range_high": decision.expected_range_high,
            "trade_allowed": decision.entry_context_valid, "reason": decision.invalid_reason,
            "model_names": decision.model_names, "model_versions": decision.model_versions,
        }
        regime = unavailable if decision is None else {
            "available": True, "regime": decision.regime, "confidence": decision.regime_confidence,
            "warnings": decision.regime_warnings,
        }
        risk = unavailable if portfolio is None else {
            "available": True, "current_equity": portfolio.current_equity,
            "realized_pnl": portfolio.realized_pnl, "unrealized_pnl": portfolio.unrealized_pnl,
            "accrued_costs": portfolio.accrued_costs,
            "margin_used": portfolio.metadata_json.get("margin_used"),
            "margin_reserved": portfolio.metadata_json.get("margin_reserved"),
            "drawdown": portfolio.metadata_json.get("drawdown"),
            "kill_switch_active": bool(kill and kill.active), "kill_switch_reason": kill.reason if kill else None,
        }
        state = {
            "schema_version": 1, "generation": generated_at, "session_id": session_id,
            "engine_action": action, "session": self._row(session), "prediction": prediction,
            "regime": regime, "risk": risk,
            "reconciliation": unavailable if reconciliation is None else self._row(reconciliation),
            "orders": [self._row(row) for row in orders], "fills": [self._row(row) for row in fills],
            "positions": [self._row(row) for row in positions], "exits": [self._row(row) for row in exits],
            "news": unavailable if decision is None else {"available": True,
                "sentiment": decision.news_sentiment, "event_risk_level": decision.event_risk_level},
            "strategy": unavailable if decision is None else {"available": True,
                "decision_cycle_id": decision.decision_cycle_id,
                "detail": decision.detail.get("strategy") if isinstance(decision.detail, dict) else None},
            "expiry_distribution": unavailable if decision is None else {"available": True,
                "expected_expiry_price": decision.expected_expiry_price, "expected_move": decision.expected_move,
                "range_68": decision.range_68, "range_95": decision.range_95,
                "tail_risk_score": decision.tail_risk_score},
            "volatility": unavailable if decision is None else {"available": True,
                "forecast": decision.volatility_forecast, "model": decision.volatility_model_used,
                "convergence_status": decision.convergence_status},
            "generated_at": generated_at,
        }
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_json(self.state_dir / "dashboard_state.json", state)
        for name, payload in {
            "prediction.json": prediction, "regime.json": regime, "risk.json": risk,
            "kill_switch.json": {"available": kill is not None, "active": bool(kill and kill.active),
                                 "reason": kill.reason if kill else None},
            "news.json": state["news"], "strategy_v2.json": state["strategy"],
            "expiry_distribution.json": state["expiry_distribution"], "vol_forecast.json": state["volatility"],
            "strategy_candidates.json": unavailable, "stress_test.json": unavailable,
            "performance_attribution.json": risk,
        }.items():
            self._atomic_json(self.state_dir / name, payload)
        self._atomic_csv(self.state_dir / "active_orders.csv",
                         (self._row(row) for row in orders if row.local_state not in {"FILLED", "CANCELLED", "REJECTED", "EXPIRED"}))
        self._atomic_csv(self.state_dir / "open_positions.csv", (self._row(row) for row in positions))
        return state

    @staticmethod
    def _row(row: Any) -> dict[str, Any]:
        return {column.name: getattr(row, column.name) for column in row.__table__.columns}

    @staticmethod
    def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, default=str)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)

    @staticmethod
    def _atomic_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
        materialized = list(rows)
        fields = sorted({key for row in materialized for key in row}) or ["id"]
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(materialized)
            handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)

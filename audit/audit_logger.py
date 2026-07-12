"""Append-only, hash-chained JSON Lines live audit trail."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
from typing import Any
from alerts.alert_manager import redact
from utils.time_utils import now_ist

EVENTS = {"market_snapshot_hash", "feature_snapshot_hash", "model_version", "regime_result", "prediction",
 "strategy_candidates", "risk_decision", "order_intent", "broker_request_id", "broker_response", "fill",
 "position_changes", "exit_decision", "reconciliation", "kill_switch_event"}

class AuditLogger:
    def __init__(self, path: str | Path = "logs/live_audit.jsonl"):
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True)
    @staticmethod
    def _digest(record: dict[str, Any]) -> str:
        return hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()
    def _last_hash(self) -> str:
        if not self.path.exists() or not self.path.stat().st_size: return "0" * 64
        return json.loads(self.path.read_text(encoding="utf-8").splitlines()[-1])["record_hash"]
    def append(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        if event_type not in EVENTS: raise ValueError(f"Unsupported audit event: {event_type}")
        record = {"timestamp": now_ist().isoformat(), "event_type": event_type,
                  "payload": redact(payload), "previous_hash": self._last_hash()}
        record["record_hash"] = self._digest(record)
        with self.path.open("a", encoding="utf-8") as stream: stream.write(json.dumps(record, sort_keys=True) + "\n")
        return record
    def verify(self) -> bool:
        previous = "0" * 64
        if not self.path.exists(): return True
        for line in self.path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line); claimed = row.pop("record_hash")
            if row["previous_hash"] != previous or self._digest(row) != claimed: return False
            previous = claimed
        return True

"""Dependency-free operator alerts with recursive secret redaction."""
from __future__ import annotations
from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Callable
from utils.time_utils import now_ist

EVENTS = {"kill_switch", "stale_data", "broker_disconnect", "failed_order", "partial_fill",
 "reconciliation_mismatch", "daily_loss_hit", "emergency_exit", "preflight_failure", "session_expiration"}
SENSITIVE = ("secret", "password", "credential", "authorization", "api_key", "token", "otp")

def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): ("***" if any(x in str(k).lower() for x in SENSITIVE) else redact(v)) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [redact(v) for v in value]
    return value

@dataclass(frozen=True)
class AlertEvent:
    event_type: str; message: str; severity: str = "ERROR"; timestamp: datetime | None = None
    context: dict[str, Any] | None = None

class AlertManager:
    def __init__(self, file_path: str | Path = "logs/operator_alerts.jsonl", *, console: bool = True,
                 email_sender: Callable[[dict[str, Any]], None] | None = None):
        self.file_path, self.console, self.email_sender = Path(file_path), console, email_sender
    def alert(self, event: AlertEvent) -> dict[str, Any]:
        if event.event_type not in EVENTS: raise ValueError(f"Unsupported alert event: {event.event_type}")
        record = redact(asdict(event)); record["timestamp"] = (event.timestamp or now_ist()).isoformat()
        line = json.dumps(record, sort_keys=True, default=str)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with self.file_path.open("a", encoding="utf-8") as stream: stream.write(line + "\n")
        if self.console: print(f"[{record['severity']}] {record['event_type']}: {record['message']}")
        if self.email_sender: self.email_sender(record)
        return record

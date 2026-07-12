"""Independent, fail-closed gate for future live order submission."""
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
import hashlib, json
from pathlib import Path
from typing import Any
from models import Mode
from utils.exceptions import ExecutionError
from utils.time_utils import now_ist

REQUIRED_CONDITIONS = ("preflight_passed", "reconciliation_passed", "kill_switch_inactive",
 "market_data_fresh", "news_risk_allowed", "risk_decision_approved", "order_validation_approved",
 "broker_session_valid", "static_ip_confirmed")

@dataclass(frozen=True)
class LiveDeploymentAcknowledgement:
    deployment_timestamp: str; approved_capital: float; maximum_lot_size: int
    enabled_symbols: tuple[str, ...]; approved_strategy_list: tuple[str, ...]
    acknowledgement_expiry: str; configuration_checksum: str

def configuration_checksum(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def validate_live_submission(*, mode: Mode, live_trading_enabled: bool,
 live_order_submission_enabled: bool, conditions: dict[str, bool], acknowledgement_path: str | Path,
 expected_checksum: str) -> LiveDeploymentAcknowledgement:
    failures = []
    if mode != Mode.LIVE: failures.append("mode == LIVE")
    if not live_trading_enabled: failures.append("live_trading_enabled")
    if not live_order_submission_enabled: failures.append("live_order_submission_enabled")
    failures += [name for name in REQUIRED_CONDITIONS if not conditions.get(name, False)]
    path = Path(acknowledgement_path)
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        ack = LiveDeploymentAcknowledgement(
            deployment_timestamp=raw["deployment_timestamp"], approved_capital=float(raw["approved_capital"]),
            maximum_lot_size=int(raw["maximum_lot_size"]), enabled_symbols=tuple(raw["enabled_symbols"]),
            approved_strategy_list=tuple(raw["approved_strategy_list"]), acknowledgement_expiry=raw["acknowledgement_expiry"],
            configuration_checksum=raw["configuration_checksum"])
        if datetime.fromisoformat(ack.acknowledgement_expiry) <= now_ist(): failures.append("acknowledgement_expiry")
        if ack.configuration_checksum != expected_checksum: failures.append("configuration_checksum")
    except Exception:
        ack = None; failures.append("manual_deployment_acknowledgement")
    if failures: raise ExecutionError("Live submission blocked: " + ", ".join(failures))
    return ack

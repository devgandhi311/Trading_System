# Autonomous Bot Roadmap

Each item is intentionally limited to roughly 3–6 files and should be delivered as a separate prompt. Complete items in dependency order. **Real order transmission remains disabled for every item below**, including live-adapter contract work; enabling transmission requires a later, separately approved final integration stage.

## Stage 1 — Backtesting trust

### Prompt 41 — Dataset provenance and legacy fail-closed loading

- **Scope:** Add dataset manifests and remove silent real-to-sample fallback from the legacy runner; sample use remains explicit.
- **Dependencies:** None.
- **Files expected:** `market/data_loader.py`, `run_backtest.py`, `models/market_models.py`, `tests/test_data_pipeline.py`, `tests/test_final_integration.py`.
- **Tests:** Missing/invalid real data fails; explicit sample succeeds; manifest includes hashes and time coverage.
- **Completion criteria:** Every backtest report identifies exact source files and generated sample is never selected implicitly.
- **Real transmission:** Disabled.

### Prompt 42 — Historical option-price ingestion

- **Scope:** Validate and normalize actual option OHLC/bid/ask/OI data with contract identity and expiry.
- **Dependencies:** Prompt 41.
- **Files expected:** `market/option_history_loader.py`, `models/market_models.py`, `config/instruments.yaml`, `tests/test_option_history_loader.py`.
- **Tests:** Duplicate tokens, missing expiry, crossed quotes, gaps and provenance.
- **Completion criteria:** Backtests can consume audited option-leg prices without synthetic substitution.
- **Real transmission:** Disabled.

### Prompt 43 — System-wide no-look-ahead invariants

- **Scope:** Convert prefix and timestamp audits into mandatory pytest invariants for features, labels, regime, predictions and fills.
- **Dependencies:** Prompt 41.
- **Files expected:** `scripts/check_lookahead_bias.py`, `tests/test_lookahead_invariants.py`, `pytest.ini`.
- **Tests:** Future-row mutation cannot change earlier outputs; split boundaries are strict.
- **Completion criteria:** Deterministic invariants pass across all decision inputs.
- **Real transmission:** Disabled.

### Prompt 44 — Unified leg-level transaction costs

- **Scope:** Make both engines use option-leg turnover and explicit charge breakdowns.
- **Dependencies:** Prompt 42.
- **Files expected:** `backtesting/transaction_costs.py`, `backtesting/backtest_engine.py`, `backtesting/backtest_engine_v2.py`, `tests/test_backtesting.py`, `tests/test_backtest_engine_v2.py`.
- **Tests:** Multi-leg, side/product charge, quantity scaling and zero-P&L independence.
- **Completion criteria:** No cost calculation uses underlying payoff or profit as turnover.
- **Real transmission:** Disabled.

## Stage 2 — Autonomous PAPER foundations

### Prompt 45 — Transactional operational database

- **Scope:** Add SQLite/SQLAlchemy schema versioning and repositories for sessions, decisions, orders, fills, positions, portfolios, reconciliation and kill-switch events.
- **Dependencies:** Prompt 41.
- **Files expected:** `storage/database.py`, `storage/models.py`, `storage/repositories.py`, `storage/__init__.py`, `tests/test_storage.py`.
- **Tests:** Commit/rollback, restart recovery, migration and no-secret persistence.
- **Completion criteria:** SQLite is authoritative; projections can be rebuilt.
- **Real transmission:** Disabled.

### Prompt 46 — Durable idempotency and unknown-order state

- **Scope:** Reserve idempotency keys transactionally and model unknown broker responses as blocking.
- **Dependencies:** Prompt 45.
- **Files expected:** `storage/repositories.py`, `models/order_models.py`, `broker/order_manager.py`, `execution/retry_handler.py`, `tests/test_order_idempotency.py`.
- **Tests:** Concurrent duplicate, restart duplicate, timeout unknown state and no blind retry.
- **Completion criteria:** One intent produces at most one submission until reconciliation resolves uncertainty.
- **Real transmission:** Disabled.

### Prompt 47 — Realistic durable PAPER broker

- **Scope:** Quote-driven limit/SL fills, latency, partial fills, fees and persistence.
- **Dependencies:** Prompts 44–46.
- **Files expected:** `broker/paper_broker.py`, `models/paper_models.py`, `storage/repositories.py`, `tests/test_paper_broker.py`.
- **Tests:** Spread/limit behavior, partial fill, costs, restart and duplicate prevention.
- **Completion criteria:** PAPER execution uses real quotes or deterministic injected quotes and survives restart.
- **Real transmission:** Disabled.

### Prompt 48 — Persistent kill switch and risk equity

- **Scope:** Persist kill state/daily state and size from reconciled current equity.
- **Dependencies:** Prompt 45.
- **Files expected:** `risk/kill_switch.py`, `risk/risk_manager.py`, `risk/position_sizing.py`, `storage/repositories.py`, `tests/test_persistent_risk.py`.
- **Tests:** Restart, daily boundary, concurrent triggers, equity drawdown sizing.
- **Completion criteria:** Restart cannot clear protection and stale/configured capital cannot replace authoritative equity.
- **Real transmission:** Disabled.

### Prompt 49 — Broker/local reconciliation service

- **Scope:** Compare orders, fills, positions, quantities and average prices; persist results and block mismatches.
- **Dependencies:** Prompts 45–47.
- **Files expected:** `runtime/reconciliation.py`, `storage/repositories.py`, `models/paper_models.py`, `tests/test_reconciliation.py`.
- **Tests:** Every mismatch, eventual consistency, unknown state and restart.
- **Completion criteria:** New entries remain blocked until a clean persisted reconciliation exists.
- **Real transmission:** Disabled.

### Prompt 50 — Autonomous PAPER engine

- **Scope:** Full once-cycle pipeline using existing features/regime/ML/strategy/risk/execution components.
- **Dependencies:** Prompts 47–49.
- **Files expected:** `runtime/paper_trading_engine.py`, `run_paper.py`, `runtime/__init__.py`, `tests/test_paper_trading_engine.py`, `tests/test_final_integration.py`.
- **Tests:** Full no-trade/trade cycles, durable decisions, stale data, restart and kill switch.
- **Completion criteria:** `--once` performs one real workflow cycle; no mock decision projections are hardcoded.
- **Real transmission:** Disabled.

### Prompt 51 — Session scheduler and continuous PAPER

- **Scope:** NSE-calendar scheduling, cutoffs, clock checks, `--continuous`, and single-cycle injection for tests.
- **Dependencies:** Prompt 50.
- **Files expected:** `runtime/session_scheduler.py`, `run_paper.py`, `utils/time_utils.py`, `tests/test_session_scheduler.py`.
- **Tests:** Holiday, market boundaries, clock drift, restart and graceful stop.
- **Completion criteria:** PAPER runs unattended only in configured sessions.
- **Real transmission:** Disabled.

### Prompt 52 — Position monitor and exit persistence

- **Scope:** Monitor all charter exit triggers and persist exit decisions before adapter calls.
- **Dependencies:** Prompts 49–51.
- **Files expected:** `runtime/position_monitor.py`, `execution/execution_engine.py`, `storage/repositories.py`, `tests/test_position_monitor.py`.
- **Tests:** Stop, target, time, expiry, regime/news/risk/kill, stale data and duplicate exit.
- **Completion criteria:** Every open PAPER position is continuously controlled and auditable.
- **Real transmission:** Disabled.

### Prompt 53 — End-of-day controller

- **Scope:** Entry cutoff, cancellations, intraday close, reconciliation, final equity, archive and unresolved report.
- **Dependencies:** Prompts 49–52.
- **Files expected:** `runtime/end_of_day.py`, `run_paper.py`, `storage/repositories.py`, `reports/daily_report.py`, `tests/test_end_of_day.py`.
- **Tests:** Cancel/close failures, unresolved state, restart and retained kill switch.
- **Completion criteria:** A full PAPER session closes or explicitly reports/blockades every unresolved item.
- **Real transmission:** Disabled.

## Stage 3 — Real-data SHADOW

### Prompt 54 — Read-only Kotak auth and instrument adapters

- **Scope:** Approved SDK wrappers for authentication and instrument master; never expose order calls.
- **Dependencies:** Secret hardening and controlled broker access.
- **Files expected:** `broker/kotak_auth.py`, `broker/kotak_read_adapter.py`, `market/instrument_master.py`, `tests/test_kotak_read_adapter.py`.
- **Tests:** Mock schemas plus separately marked broker-network contract tests.
- **Completion criteria:** Session and token master are verified read-only against Kotak.
- **Real transmission:** Disabled.

### Prompt 55 — Read-only quotes, WebSocket and live chain

- **Scope:** Real quote/WebSocket subscriptions, sequence/freshness checks and complete option-chain snapshots.
- **Dependencies:** Prompt 54.
- **Files expected:** `broker/kotak_read_adapter.py`, `broker/websocket_feed.py`, `market/option_chain.py`, `market/market_data_manager.py`, `tests/test_live_market_data_contract.py`.
- **Tests:** Reconnect, gaps, expiry, crossed/stale/missing quotes; marked network tests.
- **Completion criteria:** LIVE/SHADOW cannot use mocks or samples and fail closed on bad data.
- **Real transmission:** Disabled.

### Prompt 56 — SHADOW engine and validation report

- **Scope:** Reuse PAPER workflow with real data/news, simulated submissions/fills and subsequent-quote comparison.
- **Dependencies:** Prompts 50–55.
- **Files expected:** `runtime/shadow_live_engine.py`, `run_shadow_live.py`, `reports/shadow_validation_report.py`, `config/shadow_thresholds.yaml`, `tests/test_shadow_live.py`.
- **Tests:** No transmission, fill/slippage/latency/missed/partial/outage reporting.
- **Completion criteria:** `--once/--continuous` work and readiness never enables LIVE.
- **Real transmission:** Disabled.

## Stage 4 — LIVE preflight and dry-run

### Prompt 57 — Production environment overlays

- **Scope:** Explicit BACKTEST/PAPER/SHADOW/LIVE overlays and forbidden combinations.
- **Dependencies:** Prompts 50 and 56.
- **Files expected:** `config/environments/*.yaml`, `utils/config_loader.py`, `tests/test_environment_config.py`.
- **Tests:** Mock/sample/paper broker rejection in SHADOW/LIVE; one-lot/defined-risk defaults.
- **Completion criteria:** Environment must be explicit and LIVE cannot inherit permissive defaults.
- **Real transmission:** Disabled.

### Prompt 58 — Complete report-only LIVE preflight

- **Scope:** All charter preflight checks, static-IP confirmation, JSON/Markdown report and alerts.
- **Dependencies:** Prompts 49, 54, 55 and 57.
- **Files expected:** `runtime/live_preflight.py`, `run_live.py`, `alerts/alert_manager.py`, `tests/test_live_preflight.py`.
- **Tests:** Every missing condition blocks; reports redact secrets; no adapter order call.
- **Completion criteria:** Default invocation is preflight and produces complete evidence.
- **Real transmission:** Disabled.

### Prompt 59 — Harden Kotak dry-run order contract

- **Scope:** Golden payloads, uniform responses, durable idempotency and approval tokens without transmission.
- **Dependencies:** Prompts 45, 46, 54 and 58.
- **Files expected:** `broker/kotak_order_payload.py`, `broker/kotak_client.py`, `storage/repositories.py`, `tests/test_kotak_order_payload.py`, `tests/test_kotak_dry_run.py`.
- **Tests:** Mapping, sanitization, mode/approval gates, restart duplicates and status mapping.
- **Completion criteria:** Dry-run has future-live request/response shape and cannot be disabled.
- **Real transmission:** Disabled.

## Stage 5–6 preparation — still no transmission

### Prompt 60 — Multi-leg failure state machine

- **Scope:** Durable hedge-first state transitions and policies for every specified failure.
- **Dependencies:** Prompts 46, 49, 52 and 59.
- **Files expected:** `runtime/live_trading_engine.py`, `runtime/execution_failures.py`, `storage/repositories.py`, `alerts/alert_manager.py`, `tests/test_live_failure_injection.py`.
- **Tests:** Leg rejection, partial fill, unknown timeout, modify/cancel failure, stale quote, margin change, disconnect, expiry and stop rejection.
- **Completion criteria:** State/block/rollback/kill/alert/reconciliation outcome is deterministic for every failure.
- **Real transmission:** Disabled.

### Prompt 61 — Process lock and safe launch scripts

- **Scope:** Single-instance lock, stale PID handling and attended PAPER/SHADOW/preflight scripts.
- **Dependencies:** Prompts 51, 56 and 58.
- **Files expected:** `runtime/process_lock.py`, `scripts/start_paper.ps1`, `scripts/start_shadow.ps1`, `scripts/start_live_preflight.ps1`, `tests/test_process_lock.py`.
- **Tests:** Duplicate/stale lock, root, environment, venv and clean release.
- **Completion criteria:** No unattended LIVE start script exists; duplicate engines are blocked.
- **Real transmission:** Disabled.

### Prompt 62 — Integrated alerts and tamper-evident audit

- **Scope:** Harden locking/redaction and wire every decision hierarchy and critical event.
- **Dependencies:** Prompts 45, 50, 56 and 60.
- **Files expected:** `audit/audit_logger.py`, `alerts/alert_manager.py`, `runtime/paper_trading_engine.py`, `runtime/shadow_live_engine.py`, `tests/test_operational_audit_alerts.py`.
- **Tests:** Tamper/delete/reorder/truncate/concurrency, delivery failure and secret redaction.
- **Completion criteria:** Full cycle is reconstructable without credentials.
- **Real transmission:** Disabled.

### Prompt 63 — Operational runbooks and final readiness suite

- **Scope:** Deterministic readiness tests, network markers, deployment/rollback/outage/reconciliation/emergency/recovery runbooks and final audit.
- **Dependencies:** All prior prompts.
- **Files expected:** `tests/test_production_readiness.py`, `pytest.ini`, `docs/OPERATIONS_RUNBOOK.md`, `scripts/project_health_check.py`, `reports/risk_reports/autonomous_bot_readiness.md`.
- **Tests:** All Definition-of-Done critical paths and acceptance commands.
- **Completion criteria:** No critical item is anything other than PASS; manual broker checks and qualification periods are documented and signed off.
- **Real transmission:** Disabled. A separate human-approved final-live task would still be required.


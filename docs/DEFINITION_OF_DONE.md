# Definition of Done

The project may be described as a **finished autonomous live bot** only when every critical requirement below is implemented, independently reviewed, tested, and supported by retained evidence. A partial implementation, passing backtest, profitable result, dry run, or successful isolated live order is not sufficient.

## 1. Stage Completion

- All six stages in the [Project Charter](../PROJECT_CHARTER.md) have passed in order.
- Each stage has a dated report containing its configuration checksum, code revision, datasets, test results, failures, waivers, and reviewer approval.
- There are no open critical defects, unexplained state mismatches, or untested safety paths.

## 2. Backtesting Integrity

- Automated tests prove that features, labels, signals, fills, and position decisions cannot access future observations.
- Data provenance, timestamps, corporate/instrument changes, expiry selection, and train/validation/test boundaries are reproducible.
- Invalid or missing real data fails closed and never silently becomes sample or synthetic data.
- Fees, taxes, bid/ask spread, slippage, latency, partial fills, rejected orders, and multi-leg turnover are modeled and documented.
- Position sizing uses point-in-time current equity.
- Out-of-sample and walk-forward results are reproducible; limitations and adverse scenarios are reported.

## 3. Autonomous PAPER Qualification

- PAPER runs unattended for the approved qualification period across complete configured sessions.
- Startup, session scheduling, signal generation, risk approval, order lifecycle, monitoring, exits, end-of-day handling, reports, alerts, and restart recovery operate without manual state repair.
- Crash-recovery and duplicate-submission tests pass at every durable workflow boundary.
- PAPER uses the production workflow with only the market/broker execution boundary simulated.

## 4. SHADOW Qualification

- SHADOW consumes validated real-time spot, futures, option-chain, news, calendar, and broker-readable account data without sending orders.
- Mock quotes and sample data are technically prohibited in SHADOW.
- Expected submissions and fills are compared with subsequent bid/ask quotes, including latency, slippage, missed fills, partial fills, rejection assumptions, outages, and strategy P&L.
- The configured shadow qualification period is completed with every readiness threshold met and no unresolved critical outage or reconciliation event.
- SHADOW completion never automatically enables LIVE.

## 5. Broker and Live Preflight

- Kotak Neo authentication, session renewal/expiry, static-IP requirements, instruments, option tokens, order fields, modifications, cancellations, status queries, positions, margins, and rate limits are manually verified against the approved production account.
- Preflight checks credentials, session, calendar, clock, data freshness, WebSocket health, option chain, instrument resolution, margin, persistence, risk state, news state, kill switch, reconciliation, audit output, and adapter mode.
- Every preflight failure blocks submission and produces a sanitized report and operator alert.
- Logs, alerts, reports, database rows, exceptions, and audit records contain no credentials, session material, approval tokens, or other secrets.

## 6. Execution Safety

- LIVE submission requires all independent configuration switches, a valid non-expired deployment acknowledgement, matching configuration checksum, successful preflight and reconciliation, fresh data, valid broker session, static-IP confirmation, risk approval, and order-validation approval.
- Idempotency is durable across retries, process restarts, and concurrent attempts.
- Unknown broker responses are reconciled before any retry or new order.
- Failure-injection tests cover partial fills, leg rejection, timeouts, stale quotes, margin changes, modify/cancel failure, disconnect, session expiry, stop rejection, and emergency exit.
- Multi-leg execution establishes defined risk safely and has documented rollback, hedge, alert, kill-switch, and operator-intervention behavior for every failure state.
- Initial LIVE restrictions from the charter are enforced in code and cannot be bypassed by strategy output.

## 7. Monitoring, Exit, and Reconciliation

- Positions, orders, quantities, average prices, fills, and statuses are continuously compared with broker state.
- Any unexplained mismatch blocks new trades and requires reconciliation.
- Stop, target, time, expiry, regime, news escalation, daily/portfolio risk, kill switch, and emergency exits are monitored continuously.
- Exit decisions are persisted before submission.
- End-of-day processing blocks entries, cancels stale orders, closes required positions, verifies broker state, persists final equity, archives state, and reports unresolved items.
- Reconciliation failure leaves the kill switch active.

## 8. Persistence and Recovery

- Transactional operational storage is the authoritative source for sessions, decisions, orders, fills, positions, portfolio snapshots, reconciliation, and kill-switch events.
- Schema versioning and migrations are tested.
- Transaction rollback, abrupt termination, corrupted projection files, stale locks, and restart recovery are tested.
- Dashboard JSON files are projections only and can be rebuilt from authoritative state.
- Only one engine instance can hold the runtime lock; duplicate and stale-process cases are tested.

## 9. Auditability and Operations

- The append-only JSON Lines audit trail covers every decision-hierarchy stage, broker request/response, fill, position change, exit, reconciliation, and kill-switch event.
- Hash-chain verification detects modification, deletion, reordering, and truncation according to the documented retention design.
- Operator alerts cover every critical safety and connectivity event and are tested for delivery failure and secret redaction.
- Metrics, structured logs, daily reports, dashboard projections, backups, retention, clock synchronization, disk capacity, and restoration procedures are documented and exercised.
- Deployment, rollback, incident response, emergency square-off, credential rotation, broker outage, and manual reconciliation runbooks have been rehearsed by the operator.

## 10. Required Automated Evidence

- The complete deterministic test suite passes from a clean environment.
- Broker-network tests are separately marked and pass in the approved controlled environment.
- Tests cover autonomous loops, no-look-ahead invariants, realistic costs, current-equity sizing, freshness, token resolution, duplicate prevention, crash recovery, reconciliation, partial fills, disconnects, session expiry, kill switch, end-of-day square-off, environment separation, sample-data rejection, and every live safety switch.
- Static analysis, dependency/security scanning, configuration validation, secret scanning, and audit-chain verification pass.
- Backtest, PAPER, SHADOW, LIVE-preflight, restart, and disaster-recovery acceptance commands and their expected outputs are documented and archived.

## 11. Restricted-Live Acceptance

- Restricted LIVE begins only after written human approval of capital, symbols, strategies, lot cap, expiry, configuration checksum, and acknowledgement expiry.
- The first-live rollout remains NIFTY-only, defined-risk-only, one-lot maximum, and one-strategy maximum.
- A supervised restricted-live qualification period completes without a critical safety breach, unexplained mismatch, duplicate order, naked exposure, or loss beyond configured controls.
- Emergency square-off and kill-switch procedures are proven under supervision.

## 12. Final Sign-Off

Before the phrase **finished autonomous live bot** may be used, the final readiness audit must classify every critical subsystem as `PASS`. `WARNING`, `FAIL`, `NOT_IMPLEMENTED`, an unverified manual broker behavior, an expired approval, or an unresolved operational risk blocks that description.

Live order submission must remain disabled by default after sign-off and require an explicit, time-bounded deployment acknowledgement for each deployment.


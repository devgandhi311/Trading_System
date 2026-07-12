# Backtest V2 Correctness Fix Report

## Scope and safety

Only the four confirmed BacktestEngineV2 accounting/data-source defects were changed. Strategy thresholds, entry filters, regime detection, ML, expected-distribution logic, and signal generation were not changed. PAPER remains the configured default; no live mode was enabled and no broker order path was called.

## 1. Expired-position lifecycle

- Root cause: expired positions were pruned only after a newly executed trade, so every rejection `continue` skipped cleanup.
- Change: `_prune_expired_positions` runs at the start of every loop iteration and once at finalization. It removes matured risk, realizes net P&L, updates equity, and marks positions closed before removal.
- Proof: `test_expired_positions_close_on_rejected_day_once_and_remove_exposure` and `test_multiple_expiries_update_equity_before_later_sizing` cover rejected/no-candidate days, idempotency, stale exposure, and multiple same-day expiries.
- Old behavior: stale risk could suppress later sizing and closed P&L was represented before the position actually expired.
- Corrected behavior: positions are recorded as trades on closure, exactly once, before that day's sizing.

## 2. Explicit sample data

- Root cause: `_load_data` caught `DataValidationError` and silently substituted generated prices.
- Change: normal runs propagate missing/malformed/schema errors. Synthetic data requires `BacktestV2Config.use_sample_data=True` or CLI `--sample`; `--sample-rows` controls deterministic row count.
- Proof: `test_sample_data_requires_explicit_configuration` and `test_real_data_error_is_not_replaced_with_sample`.
- Sample status: `--sample --sample-rows 500` succeeded and printed `Data source: SAMPLE`.
- Real status: the project NSE workbook loaded successfully and printed `Data source: REAL`. Failures can no longer become sample runs.
- Reports include source type, rows, first/last dates, and source filenames.

## 3. Transaction costs

- Root cause: costs used absolute gross P&L as price with quantity 1.
- Change: each candidate leg now contributes entry premium turnover and approximated expiry intrinsic-value turnover. Brokerage is charged per entry/exit order; existing exchange/tax rates form statutory charges; existing slippage is reported separately. Multi-leg totals are aggregated.
- Proof: `test_turnover_costs_are_independent_of_profit_and_scale_with_quantity` verifies zero/profit/loss independence, proportional variable costs, and four-leg aggregation.
- Trade records include entry, exit, and total turnover; brokerage; statutory charges; slippage; total cost; gross and net P&L. `premium_source` is `APPROXIMATED`.
- Formula: `net_pnl = gross_pnl - total_transaction_cost`.

## 4. Current-equity sizing

- Root cause: optimizer capital and risk budgets always used initial capital.
- Change: `PortfolioState` tracks initial/current equity, realized/unrealized P&L, available capital, capital at risk, peak equity, drawdown, and daily realized P&L. Expiry net P&L updates equity before sizing; risk budgets use current equity. Non-positive equity blocks entry.
- Proof: the expiry/equity tests verify profits and losses change equity, costs reduce it, peak-based drawdown is retained, and closure precedes sizing. Existing optimizer tests remain green.

## Before/after sample results

Identical deterministic 500-row sample and unchanged signal logic:

| Metric | Old accounting applied to same trades | Corrected accounting |
|---|---:|---:|
| Trades | 4 | 4 |
| Gross P&L | 16,557.69 | 16,557.69 |
| Transaction costs | ~252.09 | 1,226.09 |
| Net P&L | ~16,305.60 | 15,331.60 |
| Round-trip turnover basis | Gross P&L (incorrect) | 1,022,044.34 |

The old column is reconstructed using the removed formula against the corrected run's identical trades. It is labeled approximate because the old report did not retain leg-level turnover.

## Verification

- Full suite: 163 passed, 1 existing GARCH convergence warning.
- Sample smoke: 4 trades, gross P&L 16,557.69, costs 1,226.09, net P&L 15,331.60.
- Real-data smoke (latest 500 rows): 1 trade, net P&L 4,055.00, source REAL.
- No `--all-rows` run was performed.

## Remaining limitations

- Historical option-chain premiums are unavailable, so entry premiums use the existing synthetic chain and exits use expiry intrinsic value; every trade labels this as `APPROXIMATED`.
- The existing transaction-cost configuration exposes combined exchange/tax percentages, not separate GST, SEBI, stamp-duty, clearing, and side-specific STT schedules. Those charges remain represented by the configured aggregate statutory rate rather than falsely itemized.
- V2 currently models expiry/time closure only; stop-loss, target, and manual closure hooks do not yet exist in this research engine.
- Brokerage is per order and therefore does not scale with quantity, while turnover-based statutory charges and slippage do.

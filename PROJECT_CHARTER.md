# Project Charter

## Primary Goal

Build a finished autonomous live NIFTY options-trading bot using the existing modular Python project.

The final system must:

1. Run continuously during configured NSE trading sessions.
2. Authenticate securely with Kotak Neo.
3. Receive and validate live spot, futures, option-chain, order, position, and margin data.
4. Calculate features without look-ahead bias.
5. Detect market regime.
6. Run ML and quantitative forecasts.
7. Estimate expiry distribution and volatility.
8. Generate defined-risk option strategy candidates.
9. Apply news and event-risk filters.
10. Apply current-equity-based position sizing.
11. Apply portfolio, daily, trade, margin, liquidity, slippage, and exposure limits.
12. Build LIMIT and SL-LIMIT orders.
13. Execute multi-leg orders safely.
14. Handle partial fills, rejected orders, timeouts, disconnects, session expiry, and unknown broker states.
15. Monitor positions continuously.
16. Exit on stop loss, target, time, expiry, regime change, risk limit, kill switch, or emergency.
17. Reconcile local state with broker state.
18. Recover safely after restart or crash.
19. Persist complete operational state.
20. Generate detailed reports, alerts, dashboard state, and append-only audit records.

## Mandatory Development Stages

1. Correct and trustworthy backtesting
2. Fully autonomous PAPER trading
3. SHADOW mode using real live data but no broker orders
4. LIVE preflight and broker dry-run
5. Restricted LIVE execution
6. Production hardening

Stages must be completed and evidenced in order. Passing a later-stage demonstration does not waive an earlier-stage requirement.

## Restricted Initial Live Rules

- NIFTY only.
- Defined-risk strategies only.
- One lot maximum.
- One open strategy maximum.
- No naked option selling.
- No short strangle.
- No overlapping expiry exposure.
- No new trade while any order state is unresolved.
- No live order unless all safety gates pass.
- Emergency square-off available.
- Live submission disabled by default.

## Non-Negotiable Safety Rules

- No hardcoded secrets.
- No future-data leakage.
- No silent fallback from invalid real data to sample data.
- No order without risk approval.
- No duplicate orders.
- No blind retry after unknown broker response.
- No trading with stale data.
- No trading with unresolved reconciliation mismatch.
- No trading after kill-switch activation.
- No automatic relaxation of risk limits to improve profitability.
- No direct ML-to-order path.
- No direct news-to-order path.

## Decision Hierarchy

```text
Market data
→ data validation
→ features
→ regime
→ ML/quant forecasts
→ strategy candidates
→ risk approval
→ order validation
→ execution
→ monitoring
→ exit
→ reconciliation
→ reporting
```

Every order and exit must be traceable through this hierarchy. A downstream component may reject or reduce an upstream decision, but it may not bypass a required predecessor.


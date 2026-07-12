# Repository Instructions for Coding Agents

These instructions apply to every coding agent working anywhere in this repository.

## Required Reading Order

Before planning, editing, or running task-specific commands, read the following in this order:

1. `PROJECT_CHARTER.md`
2. `docs/DEFINITION_OF_DONE.md`
3. `README.md`
4. `PROJECT_TREE.md`
5. `config/settings.yaml`
6. `config/risk_limits.yaml`
7. `config/strategy_config.yaml`
8. Relevant tests for the task

The permanent project goal is to build a finished autonomous live NIFTY options-trading bot.

Every task must advance that goal without weakening safety, correctness, testability, or auditability.

## Required Task-Start Summary

Every future Codex task must begin by summarizing:

- Current stage
- Task scope
- Files expected to change
- Safety constraints being preserved

Do not begin implementation until the required reading is complete and this summary has been provided.

## Global Coding Rules

- Work only inside the repository.
- Preserve the modular architecture.
- Prefer small changes.
- One task should address one subsystem.
- Modify related files only when integration requires it.
- Do not perform unrelated refactoring.
- Do not enable live order transmission unless the task explicitly belongs to the final approved live-integration stage.
- PAPER remains default.
- LIVE submission remains disabled by default.
- Never expose secrets.
- Use Python 3.11+.
- Use type hints, docstrings, structured logging, explicit exceptions, and deterministic tests.
- Reuse shared models, `ConfigLoader`, utilities, risk controls, and broker abstractions.
- Avoid duplicate logic.
- Avoid circular imports.
- Preserve backward compatibility where safe.
- Do not claim a module works unless tests or diagnostics verify it.
- Do not fake successful tests.
- Do not silently replace invalid production data with sample data.
- Do not change strategy thresholds while fixing infrastructure bugs.
- Do not optimize profitability during correctness tasks.
- Never use future information at decision time.
- Keep order submission idempotent.
- Treat unknown broker order state as blocking.
- Reconcile local and broker state before new trading.

## Required Completion Report for Every Task

Every task completion report must include:

- Files created
- Files modified
- Behavior changed
- Tests added
- Commands run
- Test results
- Remaining limitations
- Exact next recommended task


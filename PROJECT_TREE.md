# Project Tree

Updated 2026-07-12 from the actual repository. Generated runtime data, logs, caches, virtual environments, and report artifacts are omitted; source/report definitions are retained.

```text
.
|-- AGENTS.md
|-- AUTONOMOUS_BOT_ROADMAP.md
|-- PROJECT_CHARTER.md
|-- PROJECT_TREE.md
|-- README.md
|-- main.py
|-- requirements.txt
|-- run_backtest.py
|-- run_backtest_v2.py
|-- run_live.py
|-- run_paper.py
|-- alerts/                 (__init__.py, alert_manager.py)
|-- audit/                  (__init__.py, audit_logger.py)
|-- backtesting/            (__init__.py, backtest_engine.py, backtest_engine_v2.py,
|                            performance_metrics.py, reports.py, transaction_costs.py, walk_forward.py)
|-- broker/                 (__init__.py, kotak_auth.py, kotak_client.py,
|                            kotak_order_payload.py, order_manager.py, websocket_feed.py)
|-- config/                 (__init__.py, broker_config.yaml, instruments.yaml, news_config.yaml,
|                            quant_config.yaml, risk_limits.yaml, settings.yaml, strategy_config.yaml)
|-- dashboard/              (__init__.py, app.py)
|-- docs/                   (DEFINITION_OF_DONE.md)
|-- execution/              (__init__.py, execution_engine.py, latency_monitor.py, order_pricing.py,
|                            order_router.py, order_validator.py, retry_handler.py, squareoff.py, trade_state.py)
|-- features/               (__init__.py, feature_pipeline.py, microstructure.py, regime_features.py,
|                            returns.py, smart_money.py, volatility.py)
|-- market/                 (__init__.py, data_loader.py, data_preprocessor.py, data_store.py,
|                            instrument_master.py, market_data_manager.py, option_chain.py, test_data_loader.py)
|-- ml/                     (__init__.py, dataset_builder.py, explainability.py, model_registry.py,
|                            prediction_engine.py, train_direction_model.py, train_regime_model.py,
|                            train_signal_model.py, train_volatility_model.py, training_utils.py)
|-- models/                 (__init__.py, broker_models.py, common.py, market_models.py, order_models.py,
|                            paper_models.py, prediction_models.py, report_models.py, risk_models.py,
|                            strategy_models.py)
|-- news/                   (__init__.py, provider clients, cache/cleaner/config/fetcher/report/scheduler/
|                            sentiment/event-risk modules, test_finnhub.py)
|-- pricing/                (__init__.py, binomial_tree.py, black_scholes.py, greeks.py,
|                            implied_volatility.py, local_volatility.py, monte_carlo.py, stochastic_models.py)
|-- quant/
|   |-- attribution/        (__init__.py, performance_attribution.py)
|   |-- distribution/       (__init__.py, empirical_distribution.py, expected_move_model.py,
|   |                        expiry_distribution.py, monte_carlo_distribution.py, parametric_distribution.py)
|   |-- optimization/       (__init__.py, portfolio_risk_optimizer.py, walk_forward_strategy_eval.py)
|   |-- regime/             (__init__.py, hmm_regime_model.py, regime_scorecard.py,
|   |                        regime_transition.py)
|   |-- stress/             (__init__.py, monte_carlo_stress.py)
|   `-- volatility/         (__init__.py, garch_model.py, vol_forecaster.py, volatility_features.py)
|-- regime/                 (__init__.py, event_filter.py, liquidity_regime.py, regime_detector.py,
|                            trend_regime.py, volatility_regime.py)
|-- reports/                (__init__.py, daily_report.py, report_generator.py, strategy_v2_report.py,
|   |                        trade_report.py)
|   `-- risk_reports/       (autonomous_bot_gap_analysis.md, autonomous_bot_gap_matrix.xlsx,
|                            existing readiness/correctness reports)
|-- risk/                   (__init__.py, daily_loss_guard.py, dynamic_hedge.py, exposure_limits.py,
|                            kill_switch.py, position_sizing.py, risk_manager.py, slippage_model.py,
|                            trade_loss_guard.py)
|-- runtime/                (__init__.py, live_safety_gate.py, market_snapshot_validator.py)
|-- storage/                (__init__.py, database.py, models.py, repositories.py,
|                            schema_version.py)
|-- scripts/                (environment/data/news/look-ahead/health diagnostics and profiling scripts)
|-- strategies/             (__init__.py, base/research bases, payoff utilities, V1/V2 selectors,
|                            credit/debit/iron-condor/long-gamma/short-strangle and V2 strategy modules)
|-- tests/                  (48 deterministic pytest modules covering current implemented components)
`-- utils/                  (__init__.py, config_loader.py, env_loader.py, exceptions.py,
                             file_utils.py, logger.py, math_utils.py, time_utils.py)
```

Top-level generated/local directories currently present but intentionally not expanded: `.agents/`, `.devin/`, `.git/`, `.pytest_cache/`, `.pytest_tmp/`, `.venv/`, `__pycache__/`, `data/`, `logs/`, `test_tmp/`, and generated report directories.

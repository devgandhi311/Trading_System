# Project Tree

Generated readiness tree, excluding `.git`, virtual environments, caches, bytecode, logs, and generated report/data files.

```text
.
|-- .env.example
|-- .gitignore
|-- PROJECT_TREE.md
|-- README.md
|-- main.py
|-- requirements.txt
|-- run_backtest.py
|-- run_live.py
|-- run_paper.py
|-- backtesting/
|   |-- __init__.py
|   |-- backtest_engine.py
|   |-- performance_metrics.py
|   |-- reports.py
|   |-- transaction_costs.py
|   `-- walk_forward.py
|-- broker/
|   |-- __init__.py
|   |-- kotak_auth.py
|   |-- kotak_client.py
|   |-- order_manager.py
|   `-- websocket_feed.py
|-- config/
|   |-- __init__.py
|   |-- broker_config.yaml
|   |-- instruments.yaml
|   |-- news_config.yaml
|   |-- risk_limits.yaml
|   |-- settings.yaml
|   `-- strategy_config.yaml
|-- dashboard/
|   |-- __init__.py
|   `-- app.py
|-- data/
|   |-- features/
|   |-- live/
|   |-- model_data/
|   |-- news_cache/
|   |-- processed/
|   `-- raw/
|       |-- broker_downloads/
|       |-- nse_30_years/
|       `-- option_chain/
|-- execution/
|   |-- __init__.py
|   |-- execution_engine.py
|   |-- latency_monitor.py
|   |-- order_pricing.py
|   |-- order_router.py
|   |-- order_validator.py
|   |-- retry_handler.py
|   |-- squareoff.py
|   `-- trade_state.py
|-- features/
|   |-- __init__.py
|   |-- feature_pipeline.py
|   |-- microstructure.py
|   |-- regime_features.py
|   |-- returns.py
|   |-- smart_money.py
|   `-- volatility.py
|-- logs/
|-- market/
|   |-- __init__.py
|   |-- data_loader.py
|   |-- data_preprocessor.py
|   |-- data_store.py
|   |-- instrument_master.py
|   |-- market_data_manager.py
|   `-- option_chain.py
|-- ml/
|   |-- __init__.py
|   |-- dataset_builder.py
|   |-- explainability.py
|   |-- model_registry.py
|   |-- prediction_engine.py
|   |-- train_direction_model.py
|   |-- train_regime_model.py
|   |-- train_signal_model.py
|   |-- train_volatility_model.py
|   `-- training_utils.py
|-- models/
|   |-- __init__.py
|   |-- broker_models.py
|   |-- common.py
|   |-- market_models.py
|   |-- order_models.py
|   |-- prediction_models.py
|   |-- report_models.py
|   |-- risk_models.py
|   `-- strategy_models.py
|-- news/
|   |-- __init__.py
|   |-- alpha_vantage_client.py
|   |-- event_risk_detector.py
|   |-- finnhub_client.py
|   |-- marketaux_client.py
|   |-- news_cache.py
|   |-- news_cleaner.py
|   |-- news_config.py
|   |-- news_fetcher.py
|   |-- news_report.py
|   |-- news_scheduler.py
|   `-- news_sentiment.py
|-- pricing/
|   |-- __init__.py
|   |-- binomial_tree.py
|   |-- black_scholes.py
|   |-- greeks.py
|   |-- implied_volatility.py
|   |-- local_volatility.py
|   |-- monte_carlo.py
|   `-- stochastic_models.py
|-- regime/
|   |-- __init__.py
|   |-- event_filter.py
|   |-- liquidity_regime.py
|   |-- regime_detector.py
|   |-- trend_regime.py
|   `-- volatility_regime.py
|-- reports/
|   |-- __init__.py
|   |-- daily/
|   |-- daily_report.py
|   |-- html_reports/
|   |-- model_reports/
|   |-- report_generator.py
|   |-- risk_reports/
|   |-- trade_logs/
|   `-- trade_report.py
|-- scripts/
|   |-- check_data_files.py
|   |-- check_data_loader.py
|   |-- check_env.py
|   |-- check_finnhub.py
|   |-- check_news_integration.py
|   |-- check_regime_distribution.py
|   `-- project_health_check.py
|-- risk/
|   |-- __init__.py
|   |-- daily_loss_guard.py
|   |-- exposure_limits.py
|   |-- kill_switch.py
|   |-- position_sizing.py
|   |-- risk_manager.py
|   |-- slippage_model.py
|   `-- trade_loss_guard.py
|-- strategies/
|   |-- __init__.py
|   |-- base_strategy.py
|   |-- credit_spread.py
|   |-- debit_spread.py
|   |-- iron_condor.py
|   |-- long_gamma.py
|   |-- option_strategy_utils.py
|   |-- short_strangle.py
|   `-- strategy_selector.py
|-- tests/
|   |-- __init__.py
|   |-- test_backtesting.py
|   |-- test_config_loader.py
|   |-- test_data_pipeline.py
|   |-- test_event_risk_detector.py
|   |-- test_execution_engine.py
|   |-- test_execution_orders.py
|   |-- test_extended_pricing.py
|   |-- test_feature_pipeline.py
|   |-- test_features.py
|   |-- test_final_integration.py
|   |-- test_kotak_auth.py
|   |-- test_latency_monitor.py
|   |-- test_market_data.py
|   |-- test_market_data_manager.py
|   |-- test_microstructure_smart_money.py
|   |-- test_ml_pipeline.py
|   |-- test_models.py
|   |-- test_news_fetcher.py
|   |-- test_news_integration.py
|   |-- test_news_sentiment.py
|   |-- test_option_strategies.py
|   |-- test_order_manager.py
|   |-- test_prediction_strategy.py
|   |-- test_pricing.py
|   |-- test_regime_detector.py
|   |-- test_reporting_dashboard.py
|   |-- test_risk_manager.py
|   |-- test_utils.py
|   `-- test_websocket_feed.py
`-- utils/
    |-- __init__.py
    |-- config_loader.py
    |-- env_loader.py
    |-- exceptions.py
    |-- file_utils.py
    |-- logger.py
    |-- math_utils.py
    `-- time_utils.py
```

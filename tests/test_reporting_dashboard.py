"""Tests for reporting and dashboard modules."""

from __future__ import annotations

from datetime import date, datetime
import json

from openpyxl import load_workbook

from dashboard.app import dashboard_state, load_json, load_table
from models import OrderSide, OrderType, RegimeResult, RegimeType, RiskDecision, StrategyLeg, StrategyPlan, StrategyType
from models.prediction_models import PredictionResult
from reports import DailyReportBuilder, ReportGenerator, TRADE_REPORT_COLUMNS, TradeReportBuilder
from utils.time_utils import KOLKATA_TZ


def _trade() -> dict:
    return {
        "trade_id": "T1",
        "entry_time": datetime(2026, 1, 1, 10, 0, tzinfo=KOLKATA_TZ),
        "exit_time": datetime(2026, 1, 1, 11, 0, tzinfo=KOLKATA_TZ),
        "symbol": "NIFTY",
        "quantity": 75,
        "entry_price": 100.0,
        "exit_price": 80.0,
        "gross_pnl": 1500.0,
        "net_pnl": 1400.0,
        "strategy_type": "credit_spread",
        "regime": "Bullish low volatility",
        "metadata": {
            "expiry": "2026-01-29",
            "greeks": {"delta": 0.2},
            "iv": 18.0,
            "iv_percentile": 55.0,
            "oi_change": 1200,
            "bid_ask_spread_percent": 0.2,
            "entry_reason": "test entry",
            "exit_reason": "target",
            "improvement_note": "keep plan",
        },
    }


def _plan() -> StrategyPlan:
    return StrategyPlan(
        strategy_id="S1",
        strategy_type=StrategyType.CREDIT_SPREAD,
        symbol="NIFTY",
        created_at=datetime(2026, 1, 1, 10, 0, tzinfo=KOLKATA_TZ),
        legs=(
            StrategyLeg(
                symbol="NIFTY26000PE",
                side=OrderSide.SELL,
                quantity=75,
                order_type=OrderType.LIMIT,
                limit_price=100.0,
                trigger_price=125.0,
            ),
        ),
        confidence=0.8,
        rationale="test rationale",
        max_loss=2000.0,
    )


def test_trade_report_builder_outputs_required_columns() -> None:
    prediction = PredictionResult(
        symbol="NIFTY",
        underlying="NIFTY",
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=KOLKATA_TZ),
        predicted_direction="bullish",
        confidence=0.8,
        confidence_score=0.8,
        horizon_minutes=375,
        regime=RegimeType.TRENDING,
        bullish_probability=0.6,
        bearish_probability=0.2,
        sideways_probability=0.2,
        expected_move_percent=1.1,
    )
    risk = RiskDecision(True, datetime(2026, 1, 1, 10, 0, tzinfo=KOLKATA_TZ), "ok", risk_used=1200.0)
    regime = RegimeResult(
        symbol="NIFTY",
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=KOLKATA_TZ),
        regime=RegimeType.TRENDING,
        confidence=0.8,
        lookback_bars=20,
        regime_name="Bullish low volatility",
    )

    row = TradeReportBuilder().build_row(
        _trade(),
        strategy_plan=_plan(),
        prediction=prediction,
        risk_decision=risk,
        entry_regime=regime,
        latency_breakdown={"total": 123},
    )

    assert tuple(row.keys()) == TRADE_REPORT_COLUMNS
    assert row["trade_id"] == "T1"
    assert row["risk_allowed"] is True
    assert "bullish" in row["direction_probability"]
    assert "total" in row["latency_breakdown"]


def test_daily_report_builder_aggregates_trade_rows() -> None:
    trades = [_trade(), {**_trade(), "trade_id": "T2", "net_pnl": -500.0, "metadata": {"mistake_flag": True}}]
    summary = DailyReportBuilder(capital=100_000.0, max_daily_loss_percent=2.0).build(
        trades,
        report_date=date(2026, 1, 1),
    )

    assert summary.total_trades == 2
    assert summary.winning_trades == 1
    assert summary.losing_trades == 1
    assert summary.net_pnl == 900.0
    assert summary.strategy_wise_performance["credit_spread"] == 900.0
    assert "T2" in summary.mistakes


def test_report_generator_writes_trade_and_daily_reports(tmp_path) -> None:
    generator = ReportGenerator(trade_report_dir=tmp_path / "trades", daily_report_dir=tmp_path / "daily")
    paths = generator.generate_all([_trade()], report_date=date(2026, 1, 1), name="unit")

    assert paths.trade_excel.exists()
    assert paths.trade_csv.exists()
    assert paths.daily_excel.exists()
    workbook = load_workbook(paths.trade_excel)
    headers = [cell.value for cell in workbook["Trades"][1]]
    assert "trade_id" in headers
    assert "latency_breakdown" in headers


def test_dashboard_load_helpers_and_state(tmp_path, monkeypatch) -> None:
    live = tmp_path / "live"
    reports = tmp_path / "reports"
    live.mkdir()
    (live / "prediction.json").write_text(json.dumps({"underlying": "NIFTY"}), encoding="utf-8")
    (live / "regime.json").write_text(json.dumps({"regime_name": "Sideways low volatility"}), encoding="utf-8")
    (live / "active_orders.csv").write_text("order_id,symbol\n1,NIFTY\n", encoding="utf-8")
    monkeypatch.setattr("dashboard.app.STATE_DIR", live)
    monkeypatch.setattr("dashboard.app.REPORT_DIR", reports)

    state = dashboard_state()

    assert load_json(live / "prediction.json", {})["underlying"] == "NIFTY"
    assert not load_table(live / "active_orders.csv").empty
    assert state["prediction"]["underlying"] == "NIFTY"
    assert state["regime"]["regime_name"] == "Sideways low volatility"

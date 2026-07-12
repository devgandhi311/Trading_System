"""Streamlit dashboard for trading system monitoring."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


STATE_DIR = Path("data/live")
REPORT_DIR = Path("reports")
LOG_DIR = Path("logs")


def load_json(path: str | Path, default: Any) -> Any:
    """Load JSON file safely."""

    file_path = Path(path)
    if not file_path.exists():
        return default
    try:
        return json.loads(file_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return default


def load_table(path: str | Path) -> pd.DataFrame:
    """Load CSV or Excel table safely."""

    file_path = Path(path)
    if not file_path.exists():
        return pd.DataFrame()
    try:
        if file_path.suffix.lower() == ".csv":
            return pd.read_csv(file_path)
        if file_path.suffix.lower() in {".xlsx", ".xls"}:
            return pd.read_excel(file_path)
    except (OSError, ValueError):
        return pd.DataFrame()
    return pd.DataFrame()


def latest_file(directory: Path, pattern: str) -> Path | None:
    """Return newest matching file."""

    if not directory.exists():
        return None
    files = sorted(directory.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
    return files[0] if files else None


def dashboard_state() -> dict[str, Any]:
    """Load dashboard state from local files."""

    prediction = load_json(STATE_DIR / "prediction.json", {})
    regime = load_json(STATE_DIR / "regime.json", {})
    risk = load_json(STATE_DIR / "risk.json", {})
    orders = load_table(STATE_DIR / "active_orders.csv")
    positions = load_table(STATE_DIR / "open_positions.csv")
    latency_file = latest_file(REPORT_DIR / "risk_reports", "latency_report_*.xlsx")
    trade_file = latest_file(REPORT_DIR / "trades", "trade_report_*.csv")
    latency = load_table(latency_file) if latency_file else pd.DataFrame()
    trades = load_table(trade_file) if trade_file else pd.DataFrame()
    kill_switch = load_json(STATE_DIR / "kill_switch.json", {"active": False, "reason": ""})
    news = load_json(STATE_DIR / "news.json", {})
    strategy_v2 = load_json(STATE_DIR / "strategy_v2.json", {})
    expiry_distribution = load_json(STATE_DIR / "expiry_distribution.json", {})
    vol_forecast = load_json(STATE_DIR / "vol_forecast.json", {})
    strategy_candidates = load_json(STATE_DIR / "strategy_candidates.json", {})
    stress_test = load_json(STATE_DIR / "stress_test.json", {})
    performance_attribution = load_json(STATE_DIR / "performance_attribution.json", {})
    daily_pnl = _daily_pnl(trades)
    return {
        "prediction": prediction,
        "regime": regime,
        "risk": risk,
        "orders": orders,
        "positions": positions,
        "latency": latency,
        "trades": trades,
        "kill_switch": kill_switch,
        "news": news,
        "strategy_v2": strategy_v2,
        "expiry_distribution": expiry_distribution,
        "vol_forecast": vol_forecast,
        "strategy_candidates": strategy_candidates,
        "stress_test": stress_test,
        "performance_attribution": performance_attribution,
        "daily_pnl": daily_pnl,
    }


def run() -> None:
    """Run Streamlit dashboard."""

    import streamlit as st

    state = dashboard_state()
    st.set_page_config(page_title="NIFTY50 Trading Dashboard", layout="wide")
    st.title("NIFTY50 Trading Dashboard")

    regime = state["regime"]
    prediction = state["prediction"]
    risk = state["risk"]
    kill_switch = state["kill_switch"]
    news = state["news"]
    strategy_v2 = state["strategy_v2"]
    expiry_distribution = state["expiry_distribution"]
    vol_forecast = state["vol_forecast"]
    strategy_candidates = state["strategy_candidates"]
    stress_test = state["stress_test"]
    performance_attribution = state["performance_attribution"]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Regime", regime.get("regime_name", regime.get("regime", "unknown")))
    col2.metric("Daily P&L", f"{state['daily_pnl']:.2f}")
    col3.metric("Risk Used", risk.get("risk_used", risk.get("capital_at_risk", "n/a")))
    col4.metric("Kill Switch", "ACTIVE" if kill_switch.get("active") else "inactive")
    st.subheader("Latest News Risk")
    st.json(
        {
            "latest_news": news.get("latest_news", []),
            "sentiment_score": news.get("sentiment_score", 0.0),
            "event_risk_level": news.get("event_risk_level", "LOW"),
            "no_trade_warning": news.get("no_trade_warning", ""),
            "reason": news.get("reason", ""),
        }
    )

    st.subheader("Prediction Output")
    st.json(
        {
            "underlying": prediction.get("underlying"),
            "bullish_probability": prediction.get("bullish_probability"),
            "bearish_probability": prediction.get("bearish_probability"),
            "sideways_probability": prediction.get("sideways_probability"),
            "expected_move_percent": prediction.get("expected_move_percent"),
            "expected_value": prediction.get("expected_value"),
            "probability_of_profit": prediction.get("probability_of_profit"),
            "expected_shortfall": prediction.get("expected_shortfall"),
            "confidence_score": prediction.get("confidence_score"),
            "recommended_strategy_type": prediction.get("recommended_strategy_type"),
            "trade_allowed": prediction.get("trade_allowed"),
            "reason": prediction.get("reason"),
        }
    )

    st.subheader("Strategy V2")
    st.json(
        {
            "expected_expiry_range": {
                "expected_expiry_price": expiry_distribution.get("expected_expiry_price"),
                "lower_bound_68": expiry_distribution.get("lower_bound_68"),
                "upper_bound_68": expiry_distribution.get("upper_bound_68"),
                "lower_bound_95": expiry_distribution.get("lower_bound_95"),
                "upper_bound_95": expiry_distribution.get("upper_bound_95"),
            },
            "volatility_forecast": vol_forecast,
            "selected_strategy": strategy_v2.get("selected_strategy", "NO_TRADE"),
            "position_size_recommendation": strategy_v2.get("position_size", {}),
            "candidate_rejection_reasons": strategy_candidates.get("rejection_reasons", []),
            "news_risk": news.get("event_risk_level", "LOW"),
            "stress_test_result": stress_test,
            "performance_attribution": performance_attribution,
        }
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Open Positions")
        st.dataframe(_empty_message(state["positions"], "No open positions"), use_container_width=True)
        st.subheader("Active Orders")
        st.dataframe(_empty_message(state["orders"], "No active orders"), use_container_width=True)
    with right:
        st.subheader("Latency")
        st.dataframe(_empty_message(state["latency"], "No latency samples"), use_container_width=True)
        st.subheader("Risk Usage")
        st.json(risk or {"status": "No risk state available"})

    st.subheader("Trade Logs")
    st.dataframe(_empty_message(state["trades"], "No trade logs"), use_container_width=True)

    st.subheader("Kill Switch Status")
    st.json(kill_switch)

    st.subheader("System Logs")
    st.text(_tail_log(LOG_DIR / "system.log"))


def _daily_pnl(trades: pd.DataFrame) -> float:
    if trades.empty or "net_pnl" not in trades.columns:
        return 0.0
    return float(pd.to_numeric(trades["net_pnl"], errors="coerce").fillna(0.0).sum())


def _empty_message(dataframe: pd.DataFrame, message: str) -> pd.DataFrame:
    if dataframe.empty:
        return pd.DataFrame({"status": [message]})
    return dataframe


def _tail_log(path: Path, lines: int = 80) -> str:
    if not path.exists():
        return "No system log available."
    content = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(content[-lines:])


if __name__ == "__main__":
    run()

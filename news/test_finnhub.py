"""Manual Finnhub connectivity diagnostic.

Run:
    python news/test_finnhub.py
"""

from __future__ import annotations

from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from news.finnhub_client import FinnhubClient, requests
from utils.env_loader import load_project_env


def _redact(value: str, secret: str | None) -> str:
    """Redact an API secret from printed diagnostic text."""

    if not secret:
        return value
    return value.replace(secret, "<redacted>")


def main() -> int:
    """Check Finnhub API key, HTTP status, response time, and headlines."""

    load_project_env(PROJECT_ROOT / ".env")
    client = FinnhubClient(timeout_seconds=10)
    print(f"FINNHUB_API_KEY loaded: {'YES' if client.api_key else 'NO'}")
    if not client.api_key:
        print("Connection success: NO")
        print("Provider: Finnhub")
        print("Response status: N/A")
        print("Response time: N/A")
        return 1
    if requests is None:
        print("Connection success: NO")
        print("Provider: Finnhub")
        print("Response status: requests package missing")
        print("Response time: N/A")
        return 1

    endpoint = "/news"
    params = {"category": "general"}
    url = client.sanitized_url(endpoint, params)
    headers = {"Accept": "application/json", "User-Agent": "nifty50-trading-news/1.0"}
    print("API loaded")
    print(f"Provider: Finnhub")
    print(f"Request URL: {url}")
    print(f"Request headers: Accept=application/json, User-Agent=nifty50-trading-news/1.0")
    print(f"Timeout seconds: {client.timeout_seconds}")
    started = time.perf_counter()
    try:
        response = requests.get(
            f"{client.base_url}{endpoint}",
            params={**params, "token": client.api_key},
            headers=headers,
            timeout=client.timeout_seconds,
        )
        elapsed = time.perf_counter() - started
        print(f"Response status: {response.status_code}")
        print(f"Response time: {elapsed:.3f}s")
        if response.status_code == 429:
            print("Connection success: NO")
            print("Reason: 429 Rate limit exceeded")
            return 1
        if response.status_code in {401, 403}:
            print("Connection success: NO")
            print(f"Reason: {response.status_code} Unauthorized or invalid API key")
            return 1
        response.raise_for_status()
        articles = client.normalize_response(response.json(), endpoint="market_news")
    except Exception as exc:
        elapsed = time.perf_counter() - started
        print("Connection success: NO")
        print(f"Response time: {elapsed:.3f}s")
        print(f"Reason: {type(exc).__name__}: {_redact(str(exc), client.api_key)}")
        return 1

    if not articles:
        print("Connection success: NO")
        print("Reason: No articles returned")
        return 1
    print("Connection success")
    print("Latest headlines")
    for article in articles[:10]:
        print(f"- {article.headline} [{article.source or article.provider}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

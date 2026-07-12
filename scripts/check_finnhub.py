"""Check Finnhub connectivity without printing secrets.

Run:
    python scripts/check_finnhub.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from news.finnhub_client import FinnhubClient, FinnhubRequestError
from utils.env_loader import get_secret_status, load_project_env


def main() -> int:
    """Print safe Finnhub status and a small headline sample."""

    load_project_env(PROJECT_ROOT / ".env")
    print(f"FINNHUB key loaded: {get_secret_status('FINNHUB_API_KEY')}")

    client = FinnhubClient(timeout_seconds=10)
    try:
        status_code, articles = client.fetch_general_market_news()
    except FinnhubRequestError as exc:
        print(f"HTTP status: {exc.status_code if exc.status_code is not None else 'N/A'}")
        print("number of articles: 0")
        print(f"error: {exc}")
        return 1

    print(f"HTTP status: {status_code}")
    print(f"number of articles: {len(articles)}")
    print("first 3 headlines:")
    for article in articles[:3]:
        print(f"- {article.headline}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

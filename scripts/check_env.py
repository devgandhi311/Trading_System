"""Print safe loaded status for required environment secrets.

Run:
    python scripts/check_env.py
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env_loader import get_secret_status, load_project_env


SECRET_NAMES = (
    "KOTAK_API_KEY",
    "FINNHUB_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "MARKETAUX_API_KEY",
)


def main() -> int:
    """Load project .env and print only YES/NO secret status."""

    load_project_env(PROJECT_ROOT / ".env")
    for name in SECRET_NAMES:
        print(f"{name} loaded: {get_secret_status(name)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

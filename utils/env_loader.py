"""Shared environment loading helpers that never print secret values."""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - dependency is declared but keep imports safe.
    load_dotenv = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_project_env(env_file: str | Path = ".env") -> bool:
    """Load project .env into ``os.environ`` without overriding existing values."""

    path = Path(env_file)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if load_dotenv is not None and path.exists():
        return bool(load_dotenv(path, override=False))
    if not path.exists():
        return False
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
    return True


def get_secret_status(name: str) -> str:
    """Return YES/NO for a secret name without exposing its value."""

    load_project_env()
    return "YES" if bool(os.getenv(name)) else "NO"


def secret_loaded_status(keys: list[str] | tuple[str, ...]) -> dict[str, bool]:
    """Return true/false loaded status for secret keys without exposing values."""

    load_project_env()
    return {key: bool(os.getenv(key)) for key in keys}

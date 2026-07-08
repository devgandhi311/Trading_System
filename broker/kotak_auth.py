"""Kotak authentication adapter with in-memory session storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import os
from pathlib import Path
from typing import Any, Protocol

import pyotp

from utils.exceptions import BrokerError
from utils.time_utils import now_ist

LOGGER = logging.getLogger("broker")


@dataclass(frozen=True)
class KotakCredentials:
    """Kotak credentials loaded from environment variables only."""

    api_key: str
    user_id: str
    mobile: str
    mpin: str
    totp_secret: str


@dataclass(frozen=True)
class KotakSession:
    """In-memory Kotak session tokens."""

    access_token: str
    refresh_token: str | None
    expires_at: datetime
    user_id: str


class KotakAuthClient(Protocol):
    """Protocol for a real Kotak SDK/API adapter."""

    def login(self, credentials: KotakCredentials, totp: str) -> MappingResponse:
        """Authenticate and return session token data."""

    def validate_session(self, session: KotakSession) -> bool:
        """Return true when a broker session is still valid."""

    def refresh_session(self, session: KotakSession) -> MappingResponse:
        """Refresh a broker session and return token data."""

    def logout(self, session: KotakSession) -> None:
        """Invalidate a broker session."""


MappingResponse = dict[str, Any]


class KotakAuth:
    """Load Kotak credentials, generate TOTP, and manage in-memory sessions."""

    REQUIRED_ENV_KEYS = (
        "KOTAK_API_KEY",
        "KOTAK_USER_ID",
        "KOTAK_MOBILE",
        "KOTAK_MPIN",
        "KOTAK_TOTP_SECRET",
    )

    def __init__(
        self,
        *,
        env_file: str | Path = ".env",
        client: KotakAuthClient | None = None,
        session_ttl_minutes: int = 60,
    ) -> None:
        """Create a Kotak auth adapter.

        Tokens are kept only on this object and are never written to disk.
        """

        if session_ttl_minutes <= 0:
            raise ValueError("session_ttl_minutes must be positive")
        self.env_file = Path(env_file)
        self.client = client
        self.session_ttl = timedelta(minutes=session_ttl_minutes)
        self.credentials: KotakCredentials | None = None
        self.session: KotakSession | None = None

    def load_credentials(self) -> KotakCredentials:
        """Load Kotak credentials from .env/environment variables."""

        self._load_env_file()
        values: dict[str, str] = {}
        missing: list[str] = []
        for key in self.REQUIRED_ENV_KEYS:
            value = os.getenv(key)
            if value:
                values[key] = value
            else:
                missing.append(key)
        if missing:
            LOGGER.error("Kotak credentials missing", extra={"missing_keys": missing})
            raise BrokerError(f"Missing Kotak credential environment variables: {', '.join(missing)}")

        self.credentials = KotakCredentials(
            api_key=values["KOTAK_API_KEY"],
            user_id=values["KOTAK_USER_ID"],
            mobile=values["KOTAK_MOBILE"],
            mpin=values["KOTAK_MPIN"],
            totp_secret=values["KOTAK_TOTP_SECRET"],
        )
        LOGGER.info("Kotak credentials loaded", extra={"user_id": self.credentials.user_id})
        return self.credentials

    def generate_totp(self) -> str:
        """Generate a current TOTP code using pyotp."""

        credentials = self.credentials or self.load_credentials()
        try:
            totp = pyotp.TOTP(credentials.totp_secret).now()
        except Exception as exc:
            LOGGER.error("Kotak TOTP generation failed", extra={"user_id": credentials.user_id})
            raise BrokerError("Failed to generate Kotak TOTP") from exc
        LOGGER.info("Kotak TOTP generated", extra={"user_id": credentials.user_id})
        return totp

    def login(self) -> KotakSession:
        """Authenticate with Kotak and store session tokens in memory only."""

        credentials = self.credentials or self.load_credentials()
        totp = self.generate_totp()
        if self.client is None:
            # TODO: Wire the real Kotak SDK/API login call here when the approved
            # SDK and exact request/response syntax are available.
            LOGGER.error("Kotak login adapter unavailable", extra={"user_id": credentials.user_id})
            raise BrokerError("Kotak login adapter is not configured")
        try:
            response = self.client.login(credentials, totp)
            self.session = self._build_session(response, credentials.user_id)
        except BrokerError:
            raise
        except Exception as exc:
            LOGGER.error("Kotak login failed", extra={"user_id": credentials.user_id})
            raise BrokerError("Kotak login failed") from exc

        LOGGER.info("Kotak login succeeded", extra={"user_id": credentials.user_id})
        return self.session

    def validate_session(self) -> bool:
        """Validate the in-memory Kotak session."""

        if self.session is None:
            LOGGER.info("Kotak session validation failed: no session")
            return False
        if self.session.expires_at <= now_ist():
            LOGGER.info("Kotak session validation failed: expired", extra={"user_id": self.session.user_id})
            return False
        if self.client is None:
            LOGGER.info("Kotak session validated locally", extra={"user_id": self.session.user_id})
            return True
        try:
            is_valid = self.client.validate_session(self.session)
        except Exception as exc:
            LOGGER.error("Kotak session validation failed", extra={"user_id": self.session.user_id})
            raise BrokerError("Kotak session validation failed") from exc
        LOGGER.info(
            "Kotak session validation completed",
            extra={"user_id": self.session.user_id, "valid": is_valid},
        )
        return is_valid

    def refresh_session(self) -> KotakSession:
        """Refresh the in-memory Kotak session."""

        if self.session is None:
            raise BrokerError("Cannot refresh Kotak session before login")
        if self.client is None:
            # TODO: Wire the real Kotak SDK/API refresh call here when the approved
            # SDK and exact request/response syntax are available.
            LOGGER.error("Kotak refresh adapter unavailable", extra={"user_id": self.session.user_id})
            raise BrokerError("Kotak refresh adapter is not configured")
        try:
            response = self.client.refresh_session(self.session)
            self.session = self._build_session(response, self.session.user_id)
        except BrokerError:
            raise
        except Exception as exc:
            LOGGER.error("Kotak session refresh failed", extra={"user_id": self.session.user_id})
            raise BrokerError("Kotak session refresh failed") from exc
        LOGGER.info("Kotak session refreshed", extra={"user_id": self.session.user_id})
        return self.session

    def logout(self) -> None:
        """Logout and clear in-memory session tokens."""

        session = self.session
        if session is None:
            LOGGER.info("Kotak logout skipped: no active session")
            return
        if self.client is not None:
            try:
                self.client.logout(session)
            except Exception as exc:
                LOGGER.error("Kotak logout failed", extra={"user_id": session.user_id})
                self.session = None
                raise BrokerError("Kotak logout failed") from exc
        self.session = None
        LOGGER.info("Kotak logout completed", extra={"user_id": session.user_id})

    def _load_env_file(self) -> None:
        if not self.env_file.exists():
            return
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise BrokerError("python-dotenv is required to load Kotak .env credentials") from exc
        load_dotenv(self.env_file, override=False)

    def _build_session(self, response: MappingResponse, user_id: str) -> KotakSession:
        access_token = response.get("access_token")
        if not isinstance(access_token, str) or not access_token:
            raise BrokerError("Kotak login response missing access token")
        refresh_token = response.get("refresh_token")
        if refresh_token is not None and not isinstance(refresh_token, str):
            raise BrokerError("Kotak login response refresh token must be a string")
        expires_at = self._parse_expiry(response.get("expires_at"))
        return KotakSession(
            access_token=access_token,
            refresh_token=refresh_token,
            expires_at=expires_at,
            user_id=user_id,
        )

    def _parse_expiry(self, value: Any) -> datetime:
        if isinstance(value, datetime):
            return value if value.tzinfo else value.replace(tzinfo=now_ist().tzinfo)
        if isinstance(value, str):
            try:
                parsed = datetime.fromisoformat(value)
            except ValueError as exc:
                raise BrokerError("Kotak session expiry must be ISO datetime") from exc
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=now_ist().tzinfo)
        return now_ist() + self.session_ttl

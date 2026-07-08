from datetime import timedelta

import pytest

from broker.kotak_auth import KotakAuth, KotakCredentials, KotakSession
from utils.exceptions import BrokerError
from utils.time_utils import now_ist


class FakeKotakClient:
    def __init__(self) -> None:
        self.logged_out = False

    def login(self, credentials: KotakCredentials, totp: str) -> dict[str, object]:
        assert credentials.user_id == "user-1"
        assert len(totp) == 6
        return {
            "access_token": "access-token",
            "refresh_token": "refresh-token",
            "expires_at": now_ist() + timedelta(minutes=5),
        }

    def validate_session(self, session: KotakSession) -> bool:
        return session.access_token == "access-token"

    def refresh_session(self, session: KotakSession) -> dict[str, object]:
        return {
            "access_token": "new-access-token",
            "refresh_token": session.refresh_token,
            "expires_at": now_ist() + timedelta(minutes=10),
        }

    def logout(self, session: KotakSession) -> None:
        self.logged_out = True


def test_load_credentials_from_env_file(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "KOTAK_API_KEY=api-key",
                "KOTAK_USER_ID=user-1",
                "KOTAK_MOBILE=9999999999",
                "KOTAK_MPIN=123456",
                "KOTAK_TOTP_SECRET=JBSWY3DPEHPK3PXP",
            )
        ),
        encoding="utf-8",
    )

    auth = KotakAuth(env_file=env_file)
    credentials = auth.load_credentials()

    assert credentials.api_key == "api-key"
    assert credentials.user_id == "user-1"
    assert auth.generate_totp().isdigit()


def test_login_refresh_validate_and_logout_with_adapter(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "KOTAK_API_KEY=api-key",
                "KOTAK_USER_ID=user-1",
                "KOTAK_MOBILE=9999999999",
                "KOTAK_MPIN=123456",
                "KOTAK_TOTP_SECRET=JBSWY3DPEHPK3PXP",
            )
        ),
        encoding="utf-8",
    )
    client = FakeKotakClient()
    auth = KotakAuth(env_file=env_file, client=client)

    session = auth.login()
    assert session.access_token == "access-token"
    assert auth.validate_session()

    refreshed = auth.refresh_session()
    assert refreshed.access_token == "new-access-token"

    auth.logout()
    assert auth.session is None
    assert client.logged_out


def test_login_without_adapter_fails_closed(tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "KOTAK_API_KEY=api-key",
                "KOTAK_USER_ID=user-1",
                "KOTAK_MOBILE=9999999999",
                "KOTAK_MPIN=123456",
                "KOTAK_TOTP_SECRET=JBSWY3DPEHPK3PXP",
            )
        ),
        encoding="utf-8",
    )

    with pytest.raises(BrokerError, match="adapter is not configured"):
        KotakAuth(env_file=env_file).login()

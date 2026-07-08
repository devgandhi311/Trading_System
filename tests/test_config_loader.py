from pathlib import Path

import pytest

from utils.config_loader import ConfigError, ConfigLoader, TradingMode


def test_loads_default_config() -> None:
    config = ConfigLoader().load()

    assert config.settings.mode == TradingMode.PAPER
    assert config.settings.live_trading_enabled is False
    assert config.risk_limits.max_open_positions > 0
    assert config.strategy.allowed_strategies
    assert config.broker.broker_name == "paper"
    assert any(instrument.symbol == "NIFTY" for instrument in config.instruments)


def test_reads_secrets_from_env_file_only(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("BROKER_API_KEY=test-key\n", encoding="utf-8")

    loader = ConfigLoader(env_file=env_file)

    assert loader.get_secret("BROKER_API_KEY") == "test-key"
    assert loader.get_secret("MISSING", required=False) is None


def test_rejects_secret_like_broker_yaml_keys(tmp_path: Path) -> None:
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    source_dir = Path("config")
    for source in source_dir.glob("*.yaml"):
        (config_dir / source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    broker_file = config_dir / "broker_config.yaml"
    broker_file.write_text(
        broker_file.read_text(encoding="utf-8") + "\napi_key: unsafe\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="Secret-like broker setting"):
        ConfigLoader(config_dir=config_dir).load()

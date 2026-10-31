from pathlib import Path

import pytest
from pydantic import ValidationError

from teammates.config import load_settings, read_env_file


def test_read_env_file_parses_comments_quotes_and_blanks(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text('# c\n\nSARVAM_API_KEY="sk_abc"\nOTHER = x=y \nnoequals\n', encoding="utf-8")
    assert read_env_file(env) == {"SARVAM_API_KEY": "sk_abc", "OTHER": "x=y"}


def test_read_env_file_missing_returns_empty(tmp_path: Path) -> None:
    assert read_env_file(tmp_path / "nope") == {}


def test_load_settings_from_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("SARVAM_API_KEY=sk_file\n", encoding="utf-8")
    settings = load_settings(env, environ={})
    assert settings.sarvam_api_key.get_secret_value() == "sk_file"
    assert settings.sarvam_base_url == "https://api.sarvam.ai/v1"


def test_environment_overrides_file(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("SARVAM_API_KEY=sk_file\n", encoding="utf-8")
    settings = load_settings(env, environ={"SARVAM_API_KEY": "sk_env", "SARVAM_BASE_URL": "u"})
    assert settings.sarvam_api_key.get_secret_value() == "sk_env"
    assert settings.sarvam_base_url == "u"


@pytest.mark.parametrize("content", ["", "SARVAM_API_KEY=\n", "SARVAM_API_KEY=   \n"])
def test_missing_or_blank_key_is_rejected(tmp_path: Path, content: str) -> None:
    env = tmp_path / ".env"
    env.write_text(content, encoding="utf-8")
    with pytest.raises(ValidationError):
        load_settings(env, environ={})


def test_key_is_not_shown_in_repr(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("SARVAM_API_KEY=sk_secret\n", encoding="utf-8")
    assert "sk_secret" not in repr(load_settings(env, environ={}))


def test_redis_url_defaults_to_localhost_and_is_secret(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("SARVAM_API_KEY=k\nREDIS_URL=redis://:pw@cache:6379/1\n", encoding="utf-8")
    settings = load_settings(env, environ={})
    assert settings.redis_url.get_secret_value() == "redis://:pw@cache:6379/1"
    assert "pw" not in repr(settings)
    default = load_settings(tmp_path / "none", environ={"SARVAM_API_KEY": "k"})
    assert default.redis_url.get_secret_value() == "redis://localhost:6379/0"

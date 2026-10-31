"""Settings loaded from environment variables, with a local .env file as fallback."""

import os
from collections.abc import Mapping
from pathlib import Path

from pydantic import BaseModel, SecretStr, field_validator


class Settings(BaseModel):
    """Runtime settings. Secrets are SecretStr so they never print or log in clear."""

    sarvam_api_key: SecretStr
    sarvam_base_url: str = "https://api.sarvam.ai/v1"
    redis_url: SecretStr = SecretStr("redis://localhost:6379/0")

    @field_validator("sarvam_api_key")
    @classmethod
    def _key_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("SARVAM_API_KEY is empty; set it in .env or the environment")
        return value


def read_env_file(path: Path) -> dict[str, str]:
    """Parse KEY=VALUE lines; skip blanks and # comments; strip matching quotes."""
    if not path.exists():
        return {}
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_settings(
    env_file: Path = Path(".env"), environ: Mapping[str, str] = os.environ
) -> Settings:
    """Build Settings; real environment variables override values in the .env file."""
    merged = {**read_env_file(env_file), **environ}
    fields = {
        "sarvam_api_key": "SARVAM_API_KEY",
        "sarvam_base_url": "SARVAM_BASE_URL",
        "redis_url": "REDIS_URL",
    }
    return Settings(**{name: merged[env] for name, env in fields.items() if env in merged})

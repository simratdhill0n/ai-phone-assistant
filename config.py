import os
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Decide which .env file to load. Defaults to dev for local work.
APP_ENV = os.getenv("APP_ENV", "dev")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=f".env.{APP_ENV}",
        env_file_encoding="utf-8",
        extra="ignore",  # ignore unknown variables instead of erroring
    )

    app_env: Literal["dev", "prod"] = APP_ENV
    public_host: str
    twilio_account_sid: str
    twilio_auth_token: SecretStr


settings = Settings()
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
        extra="ignore",
    )

    app_env: Literal["dev", "prod"] = APP_ENV

    # Server
    public_host: str

    # Twilio
    twilio_account_sid: str
    twilio_auth_token: SecretStr
    twilio_phone_number: str

    # Owner and assistant
    owner_name: str
    owner_phone: str
    owner_timezone: str = "America/Toronto"
    assistant_name: str = "Nova"

    # Database
    database_url: str = "sqlite:///assistant.db"

    # LLM
    ollama_model: str = "qwen2.5:7b"

    # Speech-to-text
    whisper_model: str = "small.en"
    whisper_device: str = "cuda"
    whisper_compute_type: str = "float16"

    # Text-to-speech
    tts_engine: Literal["kokoro", "piper"] = "kokoro"
    kokoro_voice: str = "af_heart"
    kokoro_speed: float = 1.0
    kokoro_device: str = "cuda"
    piper_voice_path: str = ""

    vad_engine: Literal["silero", "energy"] = "silero"

    transfer_enabled: bool = True
    transfer_ring_seconds: int = 20


settings = Settings()
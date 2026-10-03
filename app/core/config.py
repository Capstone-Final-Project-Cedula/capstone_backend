"""
app/core/config.py
Application configuration using pydantic-settings.
Reads from .env file and environment variables.
"""
from pydantic_settings import BaseSettings
from pydantic import field_validator
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    # App
    APP_NAME: str = "CedulaSystem"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True

    # Database
    DATABASE_URL: str = "sqlite:///./cedula.db"

    # Supabase Storage (optional — PDFs and profile pictures stay local when unset)
    SUPABASE_URL: str = ""
    SUPABASE_SECRET_KEY: str = ""                  # sb_secret_... (server-side only, never in clients)
    SUPABASE_PDF_BUCKET: str = "ctc-pdfs"
    SUPABASE_PROFILE_BUCKET: str = "profile-pictures"

    # Barangay local time (Philippines is UTC+8, no DST). Used for "today",
    # monthly stats and CTC numbering so every device agrees on the date.
    LOCAL_UTC_OFFSET_HOURS: int = 8

    # JWT
    SECRET_KEY: str = "change-this-in-production-32-chars-minimum"
    JWT_SECRET_KEY: str = "jwt-secret-change-in-production"
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    # Whisper ASR
    WHISPER_MODEL: str = "small"                   # was "base" — too weak for short answers/names
    WHISPER_NO_SPEECH_THRESHOLD: float = 0.35
    WHISPER_LOGPROB_THRESHOLD: float = -1.5

    # QR Code
    QR_EXPIRY_MINUTES: int = 60

    # File paths
    UPLOAD_DIR: str = "uploads"
    PDF_OUTPUT_DIR: str = "pdfs"
    WATERMARK_PATH: str = "assets/barangay_seal.png"

    # Barangay metadata
    BARANGAY_NAME: str = "Barangay San Roque"
    MUNICIPALITY: str = "Mambajao"
    PROVINCE: str = "Camiguin"

    class Config:
        env_file = ".env"
        case_sensitive = True

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def normalize_database_url(cls, value):
        # SQLAlchemy 2.0 rejects the "postgres://" alias some providers hand out.
        if isinstance(value, str) and value.startswith("postgres://"):
            return "postgresql://" + value[len("postgres://"):]
        return value

    @field_validator("SUPABASE_URL", mode="before")
    @classmethod
    def strip_supabase_url(cls, value):
        return value.strip().rstrip("/") if isinstance(value, str) else value

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_flag(cls, value):
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"release", "production", "prod"}:
                return False
            if normalized in {"debug", "development", "dev"}:
                return True
        return value


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

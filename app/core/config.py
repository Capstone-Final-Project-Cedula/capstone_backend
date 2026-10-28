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
    # True logs every SQL statement (with citizen data), lets any website call the API
    # and opens /docs. Off unless .env turns it on, so a deploy that forgets it stays safe.
    DEBUG: bool = False

    # Database — Supabase Postgres session pooler URL, set in .env (required)
    DATABASE_URL: str

    # Supabase Storage (optional — PDFs and profile pictures stay local when unset)
    SUPABASE_URL: str = ""
    SUPABASE_SECRET_KEY: str = ""                  # sb_secret_... (server-side only, never in clients)
    SUPABASE_PDF_BUCKET: str = "ctc-pdfs"
    SUPABASE_PROFILE_BUCKET: str = "profile-pictures"
    SUPABASE_EVAL_BUCKET: str = "eval-audio"       # private; evaluation-mode clips (plan 003)

    # Web app origins allowed to call the API when DEBUG is off (comma-separated),
    # e.g. "http://localhost:3000,http://192.168.1.8:3000"
    CORS_ORIGINS: str = "http://localhost:3000"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip().rstrip("/") for o in self.CORS_ORIGINS.split(",") if o.strip()]

    # Barangay local time (Philippines is UTC+8, no DST). Used for "today",
    # monthly stats and CTC numbering so every device agrees on the date.
    LOCAL_UTC_OFFSET_HOURS: int = 8

    # JWT
    SECRET_KEY: str = "change-this-in-production-32-chars-minimum"
    JWT_SECRET_KEY: str                            # required: random secret in .env, signs all login tokens
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 480

    # Whisper ASR — speech-to-text for the live interview.
    # Tuned for short one-answer clips on an ordinary CPU server: "base" with
    # greedy decoding (~0.75 s per answer measured on an i5-11400H), versus
    # "small" + beam search at ~4 s. Use "small" on a GPU or a fast server.
    WHISPER_MODEL: str = "base"
    # Yes/no replies ("is that correct?") use a smaller model: 0.28 s instead of 0.64 s per
    # reply with the same accuracy on Philippine voices (benchmark, October 2026). "" = WHISPER_MODEL.
    WHISPER_CONFIRM_MODEL: str = "tiny"
    # One model + beam per interview question (plan 006). False = every question uses WHISPER_MODEL +
    # WHISPER_BEAM_SIZE again (for a laptop that cannot hold or run medium). Edit .env, restart.
    WHISPER_FIELD_MODELS: bool = True
    WHISPER_BEAM_SIZE: int = 1                     # 1 = greedy (fast); 5 = thorough but ~2x slower
    WHISPER_PRELOAD: bool = True                   # load at startup so the first applicant doesn't wait
    WHISPER_NO_SPEECH_THRESHOLD: float = 0.35
    WHISPER_LOGPROB_THRESHOLD: float = -1.5
    # Engine (plan 002): "openai" = openai-whisper (PyTorch), "faster" = faster-whisper (CTranslate2,
    # same models, int8 on CPU). "faster" falls back to "openai" with a log line if it is not installed.
    WHISPER_ENGINE: str = "openai"
    WHISPER_COMPUTE_TYPE: str = "int8"             # faster engine only: int8 = fastest on a CPU
    WHISPER_VAD: bool = True                       # faster engine only: skip silence so it is not turned into words

    # DistilBERT NER used to understand interview answers (names, places).
    # False skips the model (regex/gazetteer only), e.g. in the test suite.
    NER_ENABLED: bool = True

    # Citizen sign-up: True = new accounts must verify their email before they can log in.
    # Verification emails are only logged today (no email sending yet), so every backend
    # must set CITIZEN_EMAIL_VERIFICATION=False in .env / on the host, or new citizens can never log in.
    CITIZEN_EMAIL_VERIFICATION: bool = True

    # Stamped on every evaluation session so results can be grouped by build; the Builder bumps it
    # as the last build step of every later plan.
    APP_BUILD: str = "version-2.3+007"

    # QR Code
    QR_EXPIRY_MINUTES: int = 60

    # File paths
    UPLOAD_DIR: str = "uploads"
    PDF_OUTPUT_DIR: str = "pdfs"
    WATERMARK_PATH: str = "assets/barangay_seal.png"

    # Barangay metadata
    BARANGAY_NAME: str = "Barangay Imbatug"
    MUNICIPALITY: str = "Baungon"
    PROVINCE: str = "Bukidnon"

    class Config:
        env_file = ".env"
        case_sensitive = True

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def normalize_database_url(cls, value):
        # Accept the plain URL Supabase shows and pin the psycopg2 driver
        # (SQLAlchemy 2.1 would otherwise pick psycopg 3, which isn't installed;
        # 2.0 also rejects the "postgres://" alias).
        if isinstance(value, str):
            for prefix in ("postgres://", "postgresql://"):
                if value.startswith(prefix):
                    return "postgresql+psycopg2://" + value[len(prefix):]
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

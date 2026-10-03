"""
app/services/settings_service.py
System settings that admins edit from the web Settings page.

Values live in the app_settings table so every backend that shares the
database uses the same barangay details and QR expiry. Reads are cached for a
minute (PDF generation and QR creation call this on every request); saving
clears the cache so the change applies immediately on the backend that saved
it and within a minute everywhere else. Falls back to the .env values if the
row is missing.
"""
import time
from dataclasses import dataclass
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import settings as env_settings
from app.core.database import SessionLocal
from app.models.app_settings import AppSettings

CACHE_SECONDS = 60
_cache: Optional["SystemSettings"] = None
_cache_time = 0.0


@dataclass(frozen=True)
class SystemSettings:
    barangay_name: str
    municipality: str
    province: str
    qr_expiry_minutes: int


def _defaults() -> SystemSettings:
    return SystemSettings(
        barangay_name=env_settings.BARANGAY_NAME,
        municipality=env_settings.MUNICIPALITY,
        province=env_settings.PROVINCE,
        qr_expiry_minutes=env_settings.QR_EXPIRY_MINUTES,
    )


def _from_row(row: AppSettings) -> SystemSettings:
    return SystemSettings(
        barangay_name=row.barangay_name,
        municipality=row.municipality,
        province=row.province,
        qr_expiry_minutes=row.qr_expiry_minutes,
    )


def get_settings(db: Optional[Session] = None) -> SystemSettings:
    global _cache, _cache_time
    if _cache is not None and time.monotonic() - _cache_time < CACHE_SECONDS:
        return _cache

    own_session = db is None
    db = db or SessionLocal()
    try:
        row = db.get(AppSettings, 1)
        _cache = _from_row(row) if row else _defaults()
    finally:
        if own_session:
            db.close()
    _cache_time = time.monotonic()
    return _cache


def update_settings(db: Session, values: dict, updated_by: int) -> SystemSettings:
    global _cache
    row = db.get(AppSettings, 1)
    if row is None:
        row = AppSettings(id=1, **vars(_defaults()))
        db.add(row)
    for key, value in values.items():
        setattr(row, key, value)
    row.updated_by = updated_by
    db.commit()
    db.refresh(row)
    _cache = None
    return _from_row(row)

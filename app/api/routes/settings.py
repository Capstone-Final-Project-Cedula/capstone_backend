"""
app/api/routes/settings.py
System settings edited from the web Settings page.
GET /api/settings  — any signed-in staff (the header shows the barangay name)
PUT /api/settings  — admin only
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.security import get_current_user, require_admin
from app.services import settings_service
from app.services.tax_service import BASIC_TAX

router = APIRouter()


class SettingsResponse(BaseModel):
    barangay_name: str
    municipality: str
    province: str
    qr_expiry_minutes: int
    basic_community_tax: float  # fixed by RA 7160 — shown read-only


class SettingsUpdate(BaseModel):
    barangay_name: str = Field(..., min_length=2, max_length=150)
    municipality: str = Field(..., min_length=2, max_length=100)
    province: str = Field(..., min_length=2, max_length=100)
    qr_expiry_minutes: int = Field(..., ge=5, le=1440)


def _response(values: settings_service.SystemSettings) -> SettingsResponse:
    return SettingsResponse(**vars(values), basic_community_tax=BASIC_TAX)


@router.get("", response_model=SettingsResponse)
def read_settings(db: Session = Depends(get_db), current_user=Depends(get_current_user)):
    return _response(settings_service.get_settings(db))


@router.put("", response_model=SettingsResponse)
def save_settings(payload: SettingsUpdate, db: Session = Depends(get_db), admin=Depends(require_admin)):
    values = {k: v.strip() if isinstance(v, str) else v for k, v in payload.model_dump().items()}
    return _response(settings_service.update_settings(db, values, updated_by=admin.id))

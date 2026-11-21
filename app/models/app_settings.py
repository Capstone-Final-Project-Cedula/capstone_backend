"""
app/models/app_settings.py
AppSettings model — one shared row of system settings edited from the web
Settings page (barangay details shown on documents, QR code expiry).
"""
from sqlalchemy import Column, Integer, String, DateTime, ForeignKey
from sqlalchemy.sql import func
from app.core.database import Base


class AppSettings(Base):
    __tablename__ = "app_settings"

    id                = Column(Integer, primary_key=True, default=1)
    barangay_name     = Column(String(150), nullable=False)
    municipality      = Column(String(100), nullable=False)
    province          = Column(String(100), nullable=False)
    qr_expiry_minutes = Column(Integer, nullable=False, default=60)
    updated_at        = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
    updated_by        = Column(Integer, ForeignKey("staff_users.id", ondelete="SET NULL"), nullable=True)

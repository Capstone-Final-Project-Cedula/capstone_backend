"""
app/models/staff_user.py
StaffUser model — barangay personnel accounts with role-based access control.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.core.database import Base
import enum


class RoleEnum(str, enum.Enum):
    ADMIN = "admin"      # Barangay captain / IT admin — full access
    STAFF = "staff"      # Secretary / clerk — issue CTC only
    VIEWER = "viewer"    # Read-only (for auditing)


class StaffUser(Base):
    __tablename__ = "staff_users"

    id = Column(Integer, primary_key=True, index=True)

    # ── Credentials ──────────────────────────────────────────────────────────
    username = Column(String(50), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)

    # ── Profile ───────────────────────────────────────────────────────────────
    full_name = Column(String(150), nullable=False)
    position = Column(String(100), nullable=True)  # e.g. "Barangay Secretary"
    email = Column(String(150), nullable=True)

    # ── Role & Status ─────────────────────────────────────────────────────────
    role = Column(Enum(RoleEnum), default=RoleEnum.STAFF)
    is_active = Column(Boolean, default=True)

    # ── Timestamps ────────────────────────────────────────────────────────────
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    last_login = Column(DateTime(timezone=True), nullable=True)

    # ── Relationships ──────────────────────────────────────────────────────────
    issuances = relationship("CTCIssuance", back_populates="staff")

    def __repr__(self):
        return f"<StaffUser username={self.username} role={self.role}>"

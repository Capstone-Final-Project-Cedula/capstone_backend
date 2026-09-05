"""
app/models/citizen_user.py
Citizen (mobile app) user model — separate from staff users.
"""
from sqlalchemy import Column, Integer, String, Boolean, DateTime, Enum
from sqlalchemy.sql import func
from app.core.database import Base
import enum


class CitizenUser(Base):
    __tablename__ = "citizen_users"

    id                  = Column(Integer, primary_key=True, index=True)
    full_name           = Column(String(150), nullable=False)
    first_name          = Column(String(60), nullable=True)
    middle_name         = Column(String(60), nullable=True)
    last_name           = Column(String(60), nullable=True)
    username            = Column(String(50),  unique=True, nullable=False,  index=True)
    email               = Column(String(150), unique=True, nullable=False, index=True)
    profile_picture     = Column(String, nullable=True)
    hashed_password     = Column(String(255), nullable=False)
    is_active           = Column(Boolean, default=True)
    is_verified         = Column(Boolean, default=False)
    verification_token  = Column(String(64), nullable=True)
    reset_token         = Column(String(64), nullable=True)
    reset_token_expires = Column(DateTime(timezone=True), nullable=True)
    created_at          = Column(DateTime(timezone=True), server_default=func.now())
    last_login          = Column(DateTime(timezone=True), nullable=True)

    def __repr__(self):
        return f"<CitizenUser email={self.email}>"

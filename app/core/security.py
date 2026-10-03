"""
app/core/security.py
Password hashing, JWT token creation and verification.
"""
from datetime import datetime, timedelta
from typing import Optional, Union
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import get_db

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# Every token says which kind of account it belongs to. Staff and citizen
# usernames live in different tables, so without this a citizen who signs up
# with a staff member's username would be accepted as that staff member.
TOKEN_TYPE_STAFF = "staff"
TOKEN_TYPE_CITIZEN = "citizen"


# ─── Password Utilities ────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


# ─── JWT Utilities ─────────────────────────────────────────────────────────────

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (
        expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, settings.JWT_SECRET_KEY, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    try:
        payload = jwt.decode(
            token,
            settings.JWT_SECRET_KEY,
            algorithms=[settings.JWT_ALGORITHM]
        )
        return payload
    except JWTError:
        return None


# ─── FastAPI Auth Dependency ───────────────────────────────────────────────────

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
):
    """
    Dependency for protected routes.
    Decodes JWT token and returns the current staff user.
    """
    from app.models.staff_user import StaffUser

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    payload = decode_access_token(token)
    if payload is None or payload.get("type") != TOKEN_TYPE_STAFF:
        raise credentials_exception

    username: str = payload.get("sub")
    if username is None:
        raise credentials_exception

    user = db.query(StaffUser).filter(StaffUser.username == username).first()
    if user is None or not user.is_active:
        raise credentials_exception

    return user


def _role_value(user) -> str:
    return getattr(user.role, "value", user.role)


def require_admin(current_user=Depends(get_current_user)):
    """Dependency for admin-only routes."""
    if _role_value(current_user) != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_staff(current_user=Depends(get_current_user)):
    """
    Dependency for routes that create or change records (issue CTCs, claim QR
    submissions). Admins and staff pass; the read-only "viewer" role does not.
    """
    if _role_value(current_user) not in ("admin", "staff"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account is read-only.",
        )
    return current_user

"""
app/api/routes/citizen_auth.py
Full authentication API for the mobile citizen app.

Endpoints:
  POST /api/citizen/register          — create account
  POST /api/citizen/login             — get JWT token
  POST /api/citizen/verify-email      — verify email with token
  POST /api/citizen/resend-verify     — resend verification email
  POST /api/citizen/forgot-password   — request password reset
  POST /api/citizen/reset-password    — reset password with token
  GET  /api/citizen/me                — get current user profile
  PUT  /api/citizen/me                — update profile
"""
import logging
import uuid
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field, validator

from app.core.config import settings
from app.core.database import get_db
from fastapi.security import OAuth2PasswordBearer

from app.core.security import (
    hash_password, verify_password, create_access_token, TOKEN_TYPE_CITIZEN,
)
from app.models.citizen_user import CitizenUser
from app.services import storage_service

logger = logging.getLogger(__name__)

router = APIRouter()

# Citizen endpoints read the token from the Authorization header, never the
# URL (URLs end up in server and proxy logs).
citizen_token_scheme = OAuth2PasswordBearer(tokenUrl="/api/citizen/login")


# ─── Schemas ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    first_name:       str = Field(..., min_length=1, max_length=60)
    middle_name:      Optional[str] = Field(None, max_length=60)
    last_name:        str = Field(..., min_length=1, max_length=60)
    username:         str = Field(..., min_length=3, max_length=50)
    password:         str = Field(..., min_length=8)
    confirm_password: str
    profile_picture:  str = Field(..., min_length=20)
    full_name:        Optional[str] = Field(None, min_length=2, max_length=150)
    email:            Optional[EmailStr] = None

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match.')
        return v

    @validator('password')
    def password_strength(cls, v):
        return _check_password_strength(v)


def _check_password_strength(v: str) -> str:
    if not any(c.isupper() for c in v):
        raise ValueError('Password must contain at least one uppercase letter.')
    if not any(c.islower() for c in v):
        raise ValueError('Password must contain at least one lowercase letter.')
    if not any(c.isdigit() for c in v):
        raise ValueError('Password must contain at least one number.')
    if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
        raise ValueError('Password must contain at least one special character.')
    return v


class ChangePasswordRequest(BaseModel):
    old_password:     str
    new_password:     str = Field(..., min_length=8)
    confirm_password: str

    @validator('new_password')
    def new_password_strength(cls, v):
        return _check_password_strength(v)

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match.')
        return v


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=50)
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type:   str = "bearer"
    user: dict


class VerifyEmailRequest(BaseModel):
    token: str


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token:            str
    new_password:     str = Field(..., min_length=8)
    confirm_password: str

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match.')
        return v


class UpdateProfileRequest(BaseModel):
    full_name: Optional[str] = Field(None, min_length=2, max_length=150)
    username:  Optional[str] = Field(None, min_length=3, max_length=50)
    profile_picture: Optional[str] = Field(None, min_length=20)


class CitizenProfileResponse(BaseModel):
    id:          int
    full_name:   str
    first_name:  Optional[str]
    middle_name: Optional[str]
    last_name:   Optional[str]
    email:       Optional[str]
    username:    Optional[str]
    profile_picture: Optional[str]
    is_verified: bool
    created_at:  datetime

    class Config:
        from_attributes = True


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _get_citizen_from_token(token: str, db: Session) -> CitizenUser:
    """Decode JWT and return CitizenUser. Raises 401 if invalid."""
    from jose import JWTError, jwt
    from app.core.config import settings

    credentials_exc = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.JWT_SECRET_KEY, algorithms=[settings.JWT_ALGORITHM])
        subject: str = payload.get("sub")
        if not subject or payload.get("type") != TOKEN_TYPE_CITIZEN:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    user = db.query(CitizenUser).filter(
        (CitizenUser.username == subject) | (CitizenUser.email == subject)
    ).first()
    if not user or not user.is_active:
        raise credentials_exc
    return user


def _store_profile_picture(user: CitizenUser, value: Optional[str]) -> Optional[str]:
    """
    Upload a new picture to Supabase Storage and return the value to save.
    Falls back to keeping the inline image if the upload fails, so sign-up
    and profile edits never break because of Storage.
    """
    try:
        return storage_service.store_profile_picture(user.id, value)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Profile picture upload for citizen %s failed, storing inline: %s", user.id, exc)
        return value


def _send_verification_email(email: str, token: str, full_name: str):
    """
    Send a verification email. In development, prints to console.
    In production, replace with SMTP / SendGrid / Mailgun.
    """
    import logging
    logger = logging.getLogger(__name__)
    # TODO: Replace with real email sending in production
    logger.info(
        f"\n{'='*60}\n"
        f"VERIFICATION EMAIL (dev mode)\n"
        f"To: {email}\n"
        f"Name: {full_name}\n"
        f"Token: {token}\n"
        f"Link: http://localhost:8000/api/citizen/verify-email?token={token}\n"
        f"{'='*60}"
    )


def _send_reset_email(email: str, token: str):
    """Send a password reset email. Console only in development."""
    import logging
    logger = logging.getLogger(__name__)
    logger.info(
        f"\n{'='*60}\n"
        f"PASSWORD RESET EMAIL (dev mode)\n"
        f"To: {email}\n"
        f"Token: {token}\n"
        f"{'='*60}"
    )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@router.post("/register", status_code=201)
def register(
    payload: RegisterRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Register a new citizen account.
    With CITIZEN_EMAIL_VERIFICATION on, the account starts unverified and a
    verification email is sent; otherwise the citizen can sign in right away.
    """
    username = payload.username.lower().strip()
    if db.query(CitizenUser).filter(CitizenUser.username == username).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="This username is already taken.",
        )

    email = payload.email.lower().strip() if payload.email else f"{username}@cedula.local"
    if db.query(CitizenUser).filter(CitizenUser.email == email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    verification_token = secrets.token_urlsafe(32)
    name_parts = [
        payload.first_name.strip(),
        payload.middle_name.strip() if payload.middle_name else "",
        payload.last_name.strip(),
    ]
    full_name = payload.full_name.strip() if payload.full_name else " ".join(
        part for part in name_parts if part
    )

    user = CitizenUser(
        full_name=full_name,
        first_name=payload.first_name.strip(),
        middle_name=payload.middle_name.strip() if payload.middle_name else None,
        last_name=payload.last_name.strip(),
        email=email,
        username=username,
        hashed_password=hash_password(payload.password),
        verification_token=verification_token,
        is_verified=not settings.CITIZEN_EMAIL_VERIFICATION,
    )
    db.add(user)
    db.flush()  # assigns user.id, used in the storage path
    user.profile_picture = _store_profile_picture(user, payload.profile_picture)
    db.commit()
    db.refresh(user)

    if settings.CITIZEN_EMAIL_VERIFICATION:
        background_tasks.add_task(_send_verification_email, user.email, verification_token, user.full_name)
        return {
            "message": "Account created. Check your email for the verification link before signing in.",
            "username": user.username,
            "requires_verification": True,
        }

    return {
        "message": "Account created successfully. You can now sign in.",
        "username": user.username,
        "requires_verification": False,
    }


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate a citizen and return a JWT access token.
    Username and password are required.
    """
    user = db.query(CitizenUser).filter(
        CitizenUser.username == payload.username.lower().strip()
    ).first()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password.",
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Your account has been deactivated. Please contact support.",
        )

    if not user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email address before logging in. Check your inbox for the verification link.",
        )

    user.last_login = datetime.utcnow()
    db.commit()

    token = create_access_token(data={"sub": user.username, "type": TOKEN_TYPE_CITIZEN})

    return TokenResponse(
        access_token=token,
        user={
            "id":          user.id,
            "full_name":   user.full_name,
            "first_name":  user.first_name,
            "middle_name": user.middle_name,
            "last_name":   user.last_name,
            "email":       user.email,
            "username":    user.username,
            "profile_picture": storage_service.profile_picture_for_client(user.profile_picture),
            "is_verified": user.is_verified,
        },
    )


@router.post("/verify-email")
def verify_email(payload: VerifyEmailRequest, db: Session = Depends(get_db)):
    """
    Verify a citizen's email address using the token sent at registration.
    """
    user = db.query(CitizenUser).filter(
        CitizenUser.verification_token == payload.token
    ).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token.",
        )

    if user.is_verified:
        return {"message": "Your email is already verified. You can log in."}

    user.is_verified = True
    user.verification_token = None
    db.commit()

    return {"message": "Email verified successfully! You can now log in."}


@router.post("/resend-verify")
def resend_verification(
    payload: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Resend a verification email to an unverified account.
    Always returns 200 to prevent email enumeration.
    """
    user = db.query(CitizenUser).filter(
        CitizenUser.email == payload.email.lower().strip()
    ).first()

    if user and not user.is_verified:
        new_token = secrets.token_urlsafe(32)
        user.verification_token = new_token
        db.commit()
        background_tasks.add_task(
            _send_verification_email,
            email=user.email,
            token=new_token,
            full_name=user.full_name,
        )

    return {
        "message": "If an unverified account exists with that email, a new verification link has been sent."
    }


@router.post("/forgot-password")
def forgot_password(
    payload: ForgotPasswordRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
):
    """
    Request a password reset email.
    Always returns 200 to prevent email enumeration.
    """
    user = db.query(CitizenUser).filter(
        CitizenUser.email == payload.email.lower().strip()
    ).first()

    if user and user.is_active:
        reset_token = secrets.token_urlsafe(32)
        user.reset_token = reset_token
        user.reset_token_expires = datetime.utcnow() + timedelta(hours=1)
        db.commit()
        background_tasks.add_task(_send_reset_email, email=user.email, token=reset_token)

    return {
        "message": "If an account exists with that email, a password reset link has been sent."
    }


@router.post("/reset-password")
def reset_password(payload: ResetPasswordRequest, db: Session = Depends(get_db)):
    """
    Reset a citizen's password using the token from the reset email.
    Token expires after 1 hour.
    """
    user = db.query(CitizenUser).filter(
        CitizenUser.reset_token == payload.token
    ).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token.",
        )

    if user.reset_token_expires and datetime.utcnow() > user.reset_token_expires.replace(tzinfo=None):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="This reset link has expired. Please request a new one.",
        )

    user.hashed_password = hash_password(payload.new_password)
    user.reset_token = None
    user.reset_token_expires = None
    db.commit()

    return {"message": "Password reset successfully. You can now log in with your new password."}


@router.put("/me/password")
def change_my_password(
    payload: ChangePasswordRequest,
    token: str = Depends(citizen_token_scheme),
    db: Session = Depends(get_db),
):
    """Change the signed-in citizen's password (e.g. after a staff-issued temporary one)."""
    user = _get_citizen_from_token(token, db)
    if not verify_password(payload.old_password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Current password is incorrect.")
    if payload.new_password == payload.old_password:
        raise HTTPException(status_code=400, detail="New password must be different from the current one.")
    user.hashed_password = hash_password(payload.new_password)
    user.reset_token = None
    user.reset_token_expires = None
    db.commit()
    return {"message": "Password changed successfully."}


@router.get("/me", response_model=CitizenProfileResponse)
def get_my_profile(
    token: str = Depends(citizen_token_scheme),
    db: Session = Depends(get_db),
):
    """Return the currently authenticated citizen's profile."""
    user = _get_citizen_from_token(token, db)
    profile = CitizenProfileResponse.model_validate(user)
    profile.profile_picture = storage_service.profile_picture_for_client(user.profile_picture)
    return profile


@router.put("/me")
def update_profile(
    payload: UpdateProfileRequest,
    token: str = Depends(citizen_token_scheme),
    db: Session = Depends(get_db),
):
    """Update citizen's full name or username."""
    user = _get_citizen_from_token(token, db)

    if payload.full_name:
        user.full_name = payload.full_name.strip()

    if payload.username is not None:
        # Same normalisation as register/login, which lowercase the username;
        # otherwise a changed username could never be used to sign in again.
        new_username = payload.username.lower().strip()
        if len(new_username) < 3:
            raise HTTPException(status_code=400, detail="Username must be at least 3 characters.")
        if new_username != user.username:
            existing = db.query(CitizenUser).filter(
                CitizenUser.username == new_username,
                CitizenUser.id != user.id,
            ).first()
            if existing:
                raise HTTPException(status_code=409, detail="Username is already taken.")
        user.username = new_username

    old_picture = user.profile_picture
    if payload.profile_picture:
        user.profile_picture = _store_profile_picture(user, payload.profile_picture)

    db.commit()
    db.refresh(user)

    if (
        storage_service.is_storage_ref(old_picture)
        and old_picture != user.profile_picture
    ):
        storage_service.delete_object(old_picture)

    return {"message": "Profile updated.", "user": {
        "full_name": user.full_name,
        "username": user.username,
        "profile_picture": payload.profile_picture
        or storage_service.profile_picture_for_client(user.profile_picture),
    }}

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
import uuid
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr, Field, validator

from app.core.database import get_db
from app.core.security import hash_password, verify_password, create_access_token
from app.models.citizen_user import CitizenUser

router = APIRouter()


# ─── Schemas ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    full_name:        str = Field(..., min_length=2, max_length=150)
    email:            EmailStr
    password:         str = Field(..., min_length=8)
    confirm_password: str
    username:         Optional[str] = Field(None, min_length=3, max_length=50)

    @validator('confirm_password')
    def passwords_match(cls, v, values):
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match.')
        return v

    @validator('password')
    def password_strength(cls, v):
        if not any(c.isupper() for c in v):
            raise ValueError('Password must contain at least one uppercase letter.')
        if not any(c.islower() for c in v):
            raise ValueError('Password must contain at least one lowercase letter.')
        if not any(c.isdigit() for c in v):
            raise ValueError('Password must contain at least one number.')
        if not any(c in '!@#$%^&*()_+-=[]{}|;:,.<>?' for c in v):
            raise ValueError('Password must contain at least one special character.')
        return v


class LoginRequest(BaseModel):
    email:    EmailStr
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


class CitizenProfileResponse(BaseModel):
    id:          int
    full_name:   str
    email:       str
    username:    Optional[str]
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
        email: str = payload.get("sub")
        if not email:
            raise credentials_exc
    except JWTError:
        raise credentials_exc

    user = db.query(CitizenUser).filter(CitizenUser.email == email).first()
    if not user or not user.is_active:
        raise credentials_exc
    return user


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
    Sends an email verification token on success.
    """
    # Check duplicate email
    if db.query(CitizenUser).filter(CitizenUser.email == payload.email).first():
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists.",
        )

    # Check duplicate username
    if payload.username:
        if db.query(CitizenUser).filter(CitizenUser.username == payload.username).first():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="This username is already taken.",
            )

    verification_token = secrets.token_urlsafe(32)

    user = CitizenUser(
        full_name=payload.full_name.strip(),
        email=payload.email.lower().strip(),
        username=payload.username.strip() if payload.username else None,
        hashed_password=hash_password(payload.password),
        verification_token=verification_token,
        is_verified=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Send verification email in background
    background_tasks.add_task(
        _send_verification_email,
        email=user.email,
        token=verification_token,
        full_name=user.full_name,
    )

    return {
        "message": "Account created successfully. Please check your email to verify your account.",
        "email": user.email,
        "requires_verification": True,
    }


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate a citizen and return a JWT access token.
    Email must be verified before login is allowed.
    """
    user = db.query(CitizenUser).filter(
        CitizenUser.email == payload.email.lower().strip()
    ).first()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
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

    token = create_access_token(data={"sub": user.email, "type": "citizen"})

    return TokenResponse(
        access_token=token,
        user={
            "id":          user.id,
            "full_name":   user.full_name,
            "email":       user.email,
            "username":    user.username,
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


@router.get("/me", response_model=CitizenProfileResponse)
def get_my_profile(
    token: str,
    db: Session = Depends(get_db),
):
    """Return the currently authenticated citizen's profile."""
    user = _get_citizen_from_token(token, db)
    return user


@router.put("/me")
def update_profile(
    payload: UpdateProfileRequest,
    token: str,
    db: Session = Depends(get_db),
):
    """Update citizen's full name or username."""
    user = _get_citizen_from_token(token, db)

    if payload.full_name:
        user.full_name = payload.full_name.strip()

    if payload.username is not None:
        if payload.username != user.username:
            existing = db.query(CitizenUser).filter(
                CitizenUser.username == payload.username,
                CitizenUser.id != user.id,
            ).first()
            if existing:
                raise HTTPException(status_code=409, detail="Username is already taken.")
        user.username = payload.username.strip() or None

    db.commit()
    db.refresh(user)
    return {"message": "Profile updated.", "user": {
        "full_name": user.full_name,
        "username": user.username,
    }}

"""
app/api/routes/auth.py
Staff authentication endpoints.
POST /api/auth/login      — issue JWT token
POST /api/auth/register   — create new staff account (admin only)
GET  /api/auth/me         — current user profile
"""
import logging
import secrets

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.database import get_db
from app.core.security import (
    hash_password, verify_password,
    create_access_token, get_current_user, require_admin, TOKEN_TYPE_STAFF,
)
from app.models.citizen_user import CitizenUser
from app.models.staff_user import StaffUser, RoleEnum
from app.schemas.schemas import (
    LoginRequest, TokenResponse, StaffUserCreate, StaffUserResponse, ChangePasswordRequest,
    CitizenPasswordResetRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter()


@router.post("/login", response_model=TokenResponse)
def login(payload: LoginRequest, db: Session = Depends(get_db)):
    """
    Authenticate a staff member and return a JWT access token.
    """
    user = db.query(StaffUser).filter(StaffUser.username == payload.username).first()

    if not user or not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated. Contact your administrator.",
        )

    # Update last login time
    user.last_login = datetime.utcnow()
    db.commit()

    access_token = create_access_token(data={"sub": user.username, "type": TOKEN_TYPE_STAFF})

    return TokenResponse(
        access_token=access_token,
        token_type="bearer",
        user={
            "id": user.id,
            "username": user.username,
            "full_name": user.full_name,
            "role": user.role,
            "position": user.position,
        },
    )


@router.post("/register", response_model=StaffUserResponse, status_code=201)
def register_staff(
    payload: StaffUserCreate,
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
):
    """
    Create a new staff user account. Admin only.
    """
    existing = db.query(StaffUser).filter(StaffUser.username == payload.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Username '{payload.username}' is already taken.",
        )

    new_user = StaffUser(
        username=payload.username,
        hashed_password=hash_password(payload.password),
        full_name=payload.full_name,
        position=payload.position,
        email=payload.email,
        role=payload.role,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user


@router.get("/me", response_model=StaffUserResponse)
def get_me(current_user=Depends(get_current_user)):
    """Return the currently authenticated user's profile."""
    return current_user


@router.put("/me/password")
def change_password(
    payload: ChangePasswordRequest,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Allow staff to change their own password (sent in the body, never the URL)."""
    if not verify_password(payload.old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect current password.")
    if len(payload.new_password) < 8:
        raise HTTPException(status_code=400, detail="New password must be at least 8 characters.")
    if payload.new_password == payload.old_password:
        raise HTTPException(status_code=400, detail="New password must be different from the current one.")

    current_user.hashed_password = hash_password(payload.new_password)
    db.commit()
    return {"message": "Password updated successfully."}


@router.post("/citizens/reset-password")
def reset_citizen_password(
    payload: CitizenPasswordResetRequest,
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
):
    """
    Give a mobile-app citizen a one-time temporary password (admin only).
    There is no email service, so a citizen who forgot their password visits
    the barangay office; after checking their ID, the admin issues a temporary
    password here and the citizen changes it in the app (Profile > Change Password).
    """
    username = payload.username.lower().strip()
    citizen = db.query(CitizenUser).filter(CitizenUser.username == username).first()
    if not citizen:
        raise HTTPException(status_code=404, detail=f"No mobile account with username '{username}'.")

    # Meets the mobile password rules: upper, lower, digit, special, 12+ chars.
    temporary = f"Ced{secrets.token_hex(4)}!{secrets.randbelow(90) + 10}"
    citizen.hashed_password = hash_password(temporary)
    citizen.reset_token = None
    citizen.reset_token_expires = None
    db.commit()
    logger.info("Admin %s reset the password of citizen %s", admin.username, citizen.username)
    return {
        "username": citizen.username,
        "full_name": citizen.full_name,
        "temporary_password": temporary,
    }


@router.get("/staff", response_model=list[StaffUserResponse])
def list_staff(
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
):
    """List all staff accounts. Admin only."""
    return db.query(StaffUser).all()


@router.patch("/staff/{user_id}/toggle-active")
def toggle_staff_active(
    user_id: int,
    db: Session = Depends(get_db),
    admin=Depends(require_admin),
):
    """Activate or deactivate a staff account. Admin only."""
    user = db.query(StaffUser).filter(StaffUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Staff user not found.")
    if user.is_active:
        # Never let the system end up with nobody able to administer it.
        if user.id == admin.id:
            raise HTTPException(status_code=400, detail="You cannot deactivate your own account.")
        if user.role == RoleEnum.ADMIN:
            active_admins = db.query(StaffUser).filter(
                StaffUser.role == RoleEnum.ADMIN, StaffUser.is_active.is_(True)
            ).count()
            if active_admins <= 1:
                raise HTTPException(status_code=400, detail="At least one active admin is required.")
    user.is_active = not user.is_active
    db.commit()
    return {"id": user.id, "is_active": user.is_active}

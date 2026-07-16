"""
app/api/routes/auth.py
Staff authentication endpoints.
POST /api/auth/login      — issue JWT token
POST /api/auth/register   — create new staff account (admin only)
GET  /api/auth/me         — current user profile
"""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from datetime import datetime

from app.core.database import get_db
from app.core.security import (
    hash_password, verify_password,
    create_access_token, get_current_user, require_admin
)
from app.models.staff_user import StaffUser
from app.schemas.schemas import LoginRequest, TokenResponse, StaffUserCreate, StaffUserResponse

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

    access_token = create_access_token(data={"sub": user.username})

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
    old_password: str,
    new_password: str,
    db: Session = Depends(get_db),
    current_user=Depends(get_current_user),
):
    """Allow staff to change their own password."""
    if not verify_password(old_password, current_user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect current password.")
    if len(new_password) < 6:
        raise HTTPException(status_code=400, detail="New password must be at least 6 characters.")

    current_user.hashed_password = hash_password(new_password)
    db.commit()
    return {"message": "Password updated successfully."}


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
    user.is_active = not user.is_active
    db.commit()
    return {"id": user.id, "is_active": user.is_active}

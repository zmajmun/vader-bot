"""
Auth endpoints: register, login, profile.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr, field_validator
from sqlalchemy.orm import Session

from web.auth import (
    create_access_token, hash_password, verify_password, get_current_user
)
from web.models import User, get_db

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ── Schemas ────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v):
        if not v.replace("_", "").isalnum():
            raise ValueError("Username must be alphanumeric (underscores allowed)")
        if len(v) < 3 or len(v) > 32:
            raise ValueError("Username must be 3–32 characters")
        return v.lower()

    @field_validator("password")
    @classmethod
    def password_strength(cls, v):
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters")
        # bcrypt hard limit is 72 bytes — truncate silently to avoid errors
        return v[:72]


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    email: str


class ProfileResponse(BaseModel):
    id: int
    username: str
    email: str
    alpaca_paper: bool
    has_alpaca_keys: bool


class UpdateKeysRequest(BaseModel):
    alpaca_key: str
    alpaca_secret: str
    alpaca_paper: bool = True


# ── Routes ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == body.email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    if db.query(User).filter(User.username == body.username).first():
        raise HTTPException(status_code=409, detail="Username already taken")

    user = User(
        email=body.email,
        username=body.username,
        hashed_pw=hash_password(body.password),
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token, username=user.username, email=user.email)


@router.post("/login", response_model=TokenResponse)
def login(form: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    # Accept either username or email in the username field
    user = (
        db.query(User).filter(User.username == form.username.lower()).first()
        or db.query(User).filter(User.email == form.username).first()
    )
    if not user or not verify_password(form.password, user.hashed_pw):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account disabled")

    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token, username=user.username, email=user.email)


@router.get("/me", response_model=ProfileResponse)
def me(user: User = Depends(get_current_user)):
    return ProfileResponse(
        id=user.id,
        username=user.username,
        email=user.email,
        alpaca_paper=user.alpaca_paper,
        has_alpaca_keys=bool(user.alpaca_key and user.alpaca_secret),
    )


@router.post("/keys")
def save_alpaca_keys(
    body: UpdateKeysRequest,
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    user.alpaca_key = body.alpaca_key
    user.alpaca_secret = body.alpaca_secret
    user.alpaca_paper = body.alpaca_paper
    db.commit()
    mode = "paper" if body.alpaca_paper else "LIVE"
    return {"message": f"Alpaca keys saved ({mode} mode)"}

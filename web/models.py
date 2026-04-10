"""
Database models — SQLite via SQLAlchemy (zero-config, file-based).
Switch DATABASE_URL to postgres:// for production.
"""
from __future__ import annotations

import os
from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text, create_engine
)
from sqlalchemy.orm import DeclarativeBase, sessionmaker

DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./vader.db")

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {},
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"

    id            = Column(Integer, primary_key=True, index=True)
    email         = Column(String, unique=True, index=True, nullable=False)
    username      = Column(String, unique=True, index=True, nullable=False)
    hashed_pw     = Column(String, nullable=False)
    is_active     = Column(Boolean, default=True)
    created_at    = Column(DateTime, default=datetime.utcnow)

    # Alpaca credentials (stored encrypted — see auth.py)
    alpaca_key    = Column(String, nullable=True)
    alpaca_secret = Column(String, nullable=True)
    alpaca_paper  = Column(Boolean, default=True)


class TradeLog(Base):
    __tablename__ = "trade_logs"

    id          = Column(Integer, primary_key=True, index=True)
    user_id     = Column(Integer, index=True, nullable=False)
    symbol      = Column(String, nullable=False)
    direction   = Column(String, nullable=False)   # LONG | SHORT
    shares      = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=False)
    stop_price  = Column(Float, nullable=False)
    tp1_price   = Column(Float, nullable=False)
    tp2_price   = Column(Float, nullable=False)
    exit_price  = Column(Float, nullable=True)
    pnl         = Column(Float, nullable=True)
    status      = Column(String, default="open")   # open | closed | cancelled
    reason      = Column(Text, nullable=True)
    confidence  = Column(Float, nullable=True)
    opened_at   = Column(DateTime, default=datetime.utcnow)
    closed_at   = Column(DateTime, nullable=True)


def init_db():
    Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

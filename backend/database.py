"""
Database configuration and session management for the backend service.

This module exposes:
- `Base`: SQLAlchemy declarative base class.
- `engine`: SQLAlchemy engine configured from `DATABASE_URL`.
- `SessionLocal`: session factory for database access.

The default database is SQLite for ease of local development and demos.
For production, set the `DATABASE_URL` environment variable to a PostgreSQL URL.
"""

from __future__ import annotations

import os
from typing import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker


# ---------------------------------------------------------------------------
# Database configuration
# ---------------------------------------------------------------------------

# Example PostgreSQL URL:
# postgresql+psycopg2://user:password@host:port/dbname
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./sensor_data.db")

connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    # SQLite needs this for use with FastAPI in a multi-threaded context
    connect_args = {"check_same_thread": False}

engine = create_engine(DATABASE_URL, echo=False, future=True, connect_args=connect_args)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    future=True,
)

Base = declarative_base()


def get_db() -> Generator:
    """
    FastAPI dependency that yields a database session.

    Yields:
        Generator: SQLAlchemy session object.

    This ensures that each request gets its own session and that the
    session is properly closed after the request is handled.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


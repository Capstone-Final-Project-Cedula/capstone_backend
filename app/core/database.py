"""
app/core/database.py
SQLAlchemy engine, session factory, and Base class.
All models import Base from here.
"""
from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

# SQLite needs check_same_thread=False for FastAPI
connect_args = {}
if "sqlite" in settings.DATABASE_URL:
    connect_args = {"check_same_thread": False}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=settings.DEBUG,
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db():
    """
    FastAPI dependency that provides a database session per request.
    Automatically closes the session when the request is done.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables():
    """Create all tables defined in models. Called on startup."""
    from app.models import applicant, issuance, mobile_submission, staff_user  # noqa: F401
    from app.models import citizen_user  # noqa: F401  ← NEW: citizen auth table
    Base.metadata.create_all(bind=engine)
    _sync_citizen_user_columns()


def _sync_citizen_user_columns():
    """Add citizen profile columns for local SQLite databases without Alembic."""
    if "sqlite" not in settings.DATABASE_URL:
        return

    additions = {
        "first_name": "ALTER TABLE citizen_users ADD COLUMN first_name VARCHAR(60)",
        "middle_name": "ALTER TABLE citizen_users ADD COLUMN middle_name VARCHAR(60)",
        "last_name": "ALTER TABLE citizen_users ADD COLUMN last_name VARCHAR(60)",
        "profile_picture": "ALTER TABLE citizen_users ADD COLUMN profile_picture VARCHAR",
    }

    with engine.begin() as conn:
        rows = conn.execute(text("PRAGMA table_info(citizen_users)")).fetchall()
        existing = {row[1] for row in rows}
        for column_name, statement in additions.items():
            if column_name not in existing:
                conn.execute(text(statement))

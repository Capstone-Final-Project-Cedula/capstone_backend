"""
app/core/database.py
SQLAlchemy engine, session factory, and Base class.
All models import Base from here.
"""
from sqlalchemy import create_engine, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

IS_SQLITE = settings.DATABASE_URL.startswith("sqlite")
IS_POSTGRES = settings.DATABASE_URL.startswith("postgres")

# SQLite needs check_same_thread=False for FastAPI
connect_args = {}
engine_kwargs = {}
if IS_SQLITE:
    connect_args = {"check_same_thread": False}
elif IS_POSTGRES:
    # Supabase pooler: drop dead connections before use and recycle idle ones
    # before the pooler closes them.
    engine_kwargs = {"pool_pre_ping": True, "pool_recycle": 300}
    connect_args = {"connect_timeout": 10}

engine = create_engine(
    settings.DATABASE_URL,
    connect_args=connect_args,
    echo=settings.DEBUG,
    **engine_kwargs,
)

if IS_POSTGRES:
    @event.listens_for(engine, "connect")
    def _use_utc(dbapi_connection, _record):
        # Naive datetimes in the code are UTC (utcnow / CURRENT_TIMESTAMP),
        # so pin the session time zone instead of trusting the server default.
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET TIME ZONE 'UTC'")
        dbapi_connection.commit()

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
    if not IS_SQLITE:
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

"""
app/core/database.py
SQLAlchemy engine, session factory, and Base class.
All models import Base from here.

The app runs on Supabase Postgres (DATABASE_URL = session pooler string).
The schema is managed by Supabase migrations; create_all() at startup only
creates tables that are missing (e.g. in the test suite's throwaway database).
"""
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from app.core.config import settings

IS_POSTGRES = settings.DATABASE_URL.startswith("postgres")

if IS_POSTGRES:
    # Supabase pooler: drop dead connections before use and recycle idle ones
    # before the pooler closes them.
    engine_kwargs = {"pool_pre_ping": True, "pool_recycle": 300}
    connect_args = {"connect_timeout": 10}
else:
    # Only the test suite uses a non-Postgres (throwaway SQLite) database.
    engine_kwargs = {}
    connect_args = {"check_same_thread": False}

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
    """Create any missing tables defined in models. Called on startup."""
    from app.models import applicant, issuance, mobile_submission, staff_user  # noqa: F401
    from app.models import citizen_user, app_settings, sus_response, record_access_log  # noqa: F401
    Base.metadata.create_all(bind=engine)

"""
app/core/database.py
SQLAlchemy engine, session factory, and Base class.
All models import Base from here.

The app runs on Supabase Postgres (DATABASE_URL = session pooler string).
The schema is managed by Supabase migrations; create_all() at startup only
creates tables that are missing (e.g. in the test suite's throwaway database).
"""
from sqlalchemy import create_engine, event, inspect
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


DATABASE_KIND = "Supabase Postgres" if IS_POSTGRES else "SQLite (test only)"

MISSING_RECORDS_AUDIT_MIGRATION = (
    "Database is missing migration 20261018090000_records_audit.sql. Apply it in the "
    "Supabase SQL editor (see capstone_docs/TESTING-LAPTOP.md)."
)


def missing_migration_message(bind=None):
    """
    The start-up message when an existing database lacks a migration the code
    needs, or None when the schema is fine. Must run BEFORE create_tables():
    create_all() would make record_access_log without its RLS and never adds the
    print_count column. An empty database (tests) has no ctc_issuance yet: fine.
    """
    inspector = inspect(bind if bind is not None else engine)
    if not inspector.has_table("ctc_issuance"):
        return None
    columns = {column["name"] for column in inspector.get_columns("ctc_issuance")}
    if "print_count" not in columns or not inspector.has_table("record_access_log"):
        return MISSING_RECORDS_AUDIT_MIGRATION
    return None


MISSING_PRACTICE_LOG_MIGRATION = (
    "PRACTICE_LOG_ENABLED is on, but the database is missing migration 20261021090000_practice_log.sql. "
    "Apply it in the Supabase SQL editor, or turn PRACTICE_LOG_ENABLED off (see capstone_docs/TESTING-LAPTOP.md)."
)


def missing_practice_log_message(bind=None):
    """
    The start-up message when the practice log is on but its tables are missing, else None.
    Log off: never checked. Must run BEFORE create_tables() like missing_migration_message().
    """
    if not settings.PRACTICE_LOG_ENABLED:
        return None
    from app.models.practice_log import PRACTICE_LOG_TABLE_NAMES
    inspector = inspect(bind if bind is not None else engine)
    if all(inspector.has_table(name) for name in PRACTICE_LOG_TABLE_NAMES):
        return None
    return MISSING_PRACTICE_LOG_MIGRATION


MISSING_EVAL_MIGRATION = (
    "Database is missing migration 20261028090000_evaluation_mode.sql (plan 003: evaluation mode). "
    "Apply it in the Supabase SQL editor and create the private 'eval-audio' storage bucket "
    "(see capstone_docs/TESTING-LAPTOP.md)."
)


def missing_eval_message(bind=None):
    """
    The start-up message when an existing database lacks the evaluation-mode migration, else None.
    Evaluation mode is ON by default, so the check always runs; an empty database (tests) has no
    app_settings yet: fine. Must run BEFORE create_tables(): create_all() would make the eval tables
    without row level security and never adds the new app_settings columns.
    """
    from app.models.eval import EVAL_TABLE_NAMES
    inspector = inspect(bind if bind is not None else engine)
    if not inspector.has_table("app_settings"):
        return None
    columns = {column["name"] for column in inspector.get_columns("app_settings")}
    if "evaluation_mode" not in columns or not all(inspector.has_table(name) for name in EVAL_TABLE_NAMES):
        return MISSING_EVAL_MIGRATION
    return None


def create_tables():
    """Create any missing tables defined in models. Called on startup."""
    from app.models import applicant, issuance, mobile_submission, staff_user  # noqa: F401
    from app.models import citizen_user, app_settings, sus_response, record_access_log  # noqa: F401
    from app.models import practice_log, eval as eval_models  # noqa: F401
    # The temporary practice-log tables (plan 001.7) exist only through their migration:
    # create_all() would make them without row level security.
    # Same for the evaluation-mode tables (plan 003).
    tables = [t for t in Base.metadata.sorted_tables if not t.info.get("practice_log") and not t.info.get("eval")]
    Base.metadata.create_all(bind=engine, tables=tables)

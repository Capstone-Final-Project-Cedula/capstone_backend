"""
app/models/practice_log.py
TEMPORARY (plan 001.7): the practice session log. One session per practice voice
interview, one answer row per finished question, and a one-row mode setting.
The tables live in supabase/migrations/20261021090000_practice_log.sql and are
removed by supabase/practice_log_drop.sql.

They are marked info["practice_log"] so create_tables() never builds them: create_all()
would make them without row level security. They exist only through the migration
(the test suite builds them itself).
"""
from sqlalchemy import (
    JSON, Boolean, CheckConstraint, Column, DateTime, Float, ForeignKey, Integer, String,
    Text, UniqueConstraint,
)
from sqlalchemy.sql import func

from app.core.database import Base

PRACTICE_MODES = ("test", "real")
PRACTICE_OUTCOMES = ("open", "issued", "abandoned")
PRACTICE_ENGINES = ("whisper", "browser", "typed", "spelled", "skipped")
PRACTICE_LOG_TABLE_NAMES = ("practice_log_settings", "practice_log_sessions", "practice_log_answers")


class PracticeLogSettings(Base):
    __tablename__ = "practice_log_settings"
    __table_args__ = {"info": {"practice_log": True}}

    id = Column(Integer, primary_key=True)   # always 1
    mode = Column(String(10), nullable=False, default="test")


class PracticeLogSession(Base):
    __tablename__ = "practice_log_sessions"
    __table_args__ = {"info": {"practice_log": True}}

    id = Column(Integer, primary_key=True, index=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    build_label = Column(String(60), nullable=True)
    backend_commit = Column(String(40), nullable=True)
    mode = Column(String(10), nullable=False)           # one of PRACTICE_MODES
    staff_user_id = Column(Integer, ForeignKey("staff_users.id", ondelete="SET NULL"), nullable=True)
    input_method = Column(String(10), nullable=False, default="voice")
    outcome = Column(String(10), nullable=False, default="open")   # one of PRACTICE_OUTCOMES
    issuance_id = Column(Integer, ForeignKey("ctc_issuance.id", ondelete="SET NULL"), nullable=True)


class PracticeLogAnswer(Base):
    __tablename__ = "practice_log_answers"
    __table_args__ = (
        UniqueConstraint("session_id", "field"),
        CheckConstraint("attempts >= 0"),
        {"info": {"practice_log": True}},
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("practice_log_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    field = Column(String(40), nullable=False)
    attempts = Column(Integer, nullable=False, default=1)
    attempt_details = Column(JSON, nullable=False, default=list)
    first_value = Column(Text, nullable=True)           # first voice-parsed value, in the saved form
    accepted_value = Column(Text, nullable=True)
    accepted_engine = Column(String(10), nullable=False)    # one of PRACTICE_ENGINES
    seconds = Column(Float, nullable=True)
    final_value = Column(Text, nullable=True)           # filled when the CTC is issued
    changed_after_voice = Column(Boolean, nullable=True)


def practice_log_tables():
    """The three practice-log tables, for tests that build them without the migration."""
    return [Base.metadata.tables[name] for name in PRACTICE_LOG_TABLE_NAMES]

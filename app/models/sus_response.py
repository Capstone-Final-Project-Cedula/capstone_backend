"""
app/models/sus_response.py
SusResponse model — one System Usability Scale (SUS) survey answer from the
web or mobile app (capstone research). The table, its CHECK rules and the
generated sus_score live in the Supabase migration
supabase/migrations/20261003150000_create_sus_responses.sql.
"""
import uuid

from sqlalchemy import (
    JSON, Boolean, Column, Computed, DateTime, ForeignKey, Integer, Numeric,
    SmallInteger, Text, Uuid,
)
from sqlalchemy.dialects.postgresql import ARRAY
from sqlalchemy.sql import func

from app.core.database import Base

# Standard SUS scoring: positive (odd) items score q-1, negative (even) items
# 5-q; the 0..40 sum times 2.5 gives a 0..100 score.
SUS_SCORE_SQL = (
    "((q1 - 1) + (5 - q2) + (q3 - 1) + (5 - q4) + (q5 - 1)"
    " + (5 - q6) + (q7 - 1) + (5 - q8) + (q9 - 1) + (5 - q10)) * 2.5"
)


class SusResponse(Base):
    __tablename__ = "sus_responses"

    id                  = Column(Uuid, primary_key=True, default=uuid.uuid4)
    created_at          = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    respondent_name     = Column(Text, nullable=True)
    role                = Column(Text, nullable=False)
    computer_experience = Column(Text, nullable=False)
    # text[] on Postgres; JSON in the test suite's SQLite database
    features_used       = Column(ARRAY(Text).with_variant(JSON(), "sqlite"), nullable=False, default=list)
    q1  = Column(SmallInteger, nullable=False)
    q2  = Column(SmallInteger, nullable=False)
    q3  = Column(SmallInteger, nullable=False)
    q4  = Column(SmallInteger, nullable=False)
    q5  = Column(SmallInteger, nullable=False)
    q6  = Column(SmallInteger, nullable=False)
    q7  = Column(SmallInteger, nullable=False)
    q8  = Column(SmallInteger, nullable=False)
    q9  = Column(SmallInteger, nullable=False)
    q10 = Column(SmallInteger, nullable=False)
    sus_score           = Column(Numeric, Computed(SUS_SCORE_SQL, persisted=True))
    comments            = Column(Text, nullable=True)
    consent_given       = Column(Boolean, nullable=False)
    platform            = Column(Text, nullable=False)
    transaction_id      = Column(Integer, ForeignKey("ctc_issuance.id", ondelete="SET NULL"), nullable=True)
    synced_to_sheets    = Column(Boolean, nullable=False, default=False)

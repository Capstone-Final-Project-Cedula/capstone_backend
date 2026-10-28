"""
app/models/eval.py
Evaluation mode (plan 003): one session per logged interview, and answer rows with what the
system heard, understood and finally confirmed, so Chapter 4's numbers come from the system itself.

`eval_answers` holds two kinds of rows, told apart by `step`:
  - one row per TRY (step spoken / spelled / end_said / confirm / review): the transcript, the model
    and beam, the Whisper seconds, the saved clip, and what /extraction/answer made of it;
  - one row per FIELD with step "final", written when the session is scored: the final confirmed
    value, how the field finished (`finished_in`), the three correct flags, the WER counts, seconds.
Numbers live in booleans and counts, so "Delete evaluation audio now" can erase every clip and every
text column and the study numbers stay.

The tables exist only through supabase/migrations/20261028090000_evaluation_mode.sql (RLS on):
they are marked info["eval"] so create_tables() never builds them with create_all (the test suite
builds them itself).
"""
from sqlalchemy import (
    Boolean, CheckConstraint, Column, DateTime, Float, ForeignKey, Integer, String, Text, UniqueConstraint,
)
from sqlalchemy.sql import func

from app.core.database import Base

ROUND_TYPES = ("practice", "scripted", "uat", "live")
ROUND_CONDITIONS = ("quiet", "noisy")
SESSION_OUTCOMES = ("open", "issued", "abandoned")
ATTEMPT_STEPS = ("spoken", "spelled", "end_said", "confirm", "review")
FINISHED_IN = (
    "spoken_1", "spoken_2", "spoken_3", "spelled_1", "spelled_2",
    "end_said", "end_typed", "staff_typed", "blank",
)
EVAL_TABLE_NAMES = ("eval_sessions", "eval_answers", "security_events")


class EvalSession(Base):
    __tablename__ = "eval_sessions"
    __table_args__ = (
        UniqueConstraint("round_no", "participant_no"),
        {"info": {"eval": True}},
    )

    id = Column(Integer, primary_key=True, index=True)
    started_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    ended_at = Column(DateTime(timezone=True), nullable=True)
    round_no = Column(Integer, nullable=False, default=1)        # bumps when the admin starts a new round
    round_type = Column(String(10), nullable=False)               # one of ROUND_TYPES
    round_condition = Column(String(10), nullable=False)          # one of ROUND_CONDITIONS
    card_set = Column(String(40), nullable=True)                  # scripted rounds only
    participant_no = Column(Integer, nullable=False)
    participant_code = Column(String(12), nullable=False)         # P001 ... (never a name)
    card_no = Column(Integer, nullable=True)                      # scripted rounds: derived from participant_no
    input_method = Column(String(10), nullable=False, default="voice")
    device = Column(String(200), nullable=True)
    build_label = Column(String(60), nullable=True)
    staff_user_id = Column(Integer, ForeignKey("staff_users.id", ondelete="SET NULL"), nullable=True)
    outcome = Column(String(10), nullable=False, default="open")  # one of SESSION_OUTCOMES
    issuance_id = Column(Integer, ForeignKey("ctc_issuance.id", ondelete="SET NULL"), nullable=True)
    error_count = Column(Integer, nullable=False, default=0)
    app_seconds = Column(Float, nullable=True)                    # consent -> CTC issued
    qr_seconds = Column(Float, nullable=True)                     # QR scan -> CTC issued (QR sessions)
    scored = Column(Boolean, nullable=False, default=False)


class EvalAnswer(Base):
    __tablename__ = "eval_answers"
    __table_args__ = (
        CheckConstraint("attempt_no >= 0"),
        UniqueConstraint("session_id", "try_id"),      # one row per try; NULL try ids (final rows) never clash
        {"info": {"eval": True}},
    )

    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey("eval_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    field = Column(String(40), nullable=False)
    step = Column(String(10), nullable=False)                     # ATTEMPT_STEPS or "final"
    attempt_no = Column(Integer, nullable=False, default=1)       # within the field and step
    try_id = Column(String(40), nullable=True)                    # browser id shared by transcribe and extraction
    engine = Column(String(10), nullable=True)                    # whisper / browser / typed
    model = Column(String(40), nullable=True)
    beam = Column(Integer, nullable=True)
    audio_ref = Column(Text, nullable=True)                       # storage reference of the saved clip
    transcript = Column(Text, nullable=True)                      # what Whisper heard (server side)
    whisper_seconds = Column(Float, nullable=True)
    extracted_value = Column(Text, nullable=True)                 # what /extraction/answer made of it
    extract_ok = Column(Boolean, nullable=True)
    yes_no = Column(String(10), nullable=True)                    # confirm / review steps: yes, no, other
    # ── the "final" row of a field ──
    expected_value = Column(Text, nullable=True)                  # the card's value (scripted rounds)
    final_value = Column(Text, nullable=True)
    finished_in = Column(String(12), nullable=True)               # one of FINISHED_IN
    spoken_tries = Column(Integer, nullable=True)
    spelled_tries = Column(Integer, nullable=True)
    skipped = Column(Boolean, nullable=True)
    correct_first = Column(Boolean, nullable=True)
    correct_ladder = Column(Boolean, nullable=True)
    correct_finished = Column(Boolean, nullable=True)
    wer_errors = Column(Integer, nullable=True)
    wer_words = Column(Integer, nullable=True)
    seconds = Column(Float, nullable=True)                        # question asked -> answer accepted


class SecurityEvent(Base):
    """Failed logins and 401/403 answers: time, route and kind only (never a password or a name)."""
    __tablename__ = "security_events"
    __table_args__ = {"info": {"eval": True}}

    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    route = Column(String(120), nullable=False)
    kind = Column(String(20), nullable=False)    # login_failed / unauthorized / forbidden


def eval_tables():
    """The evaluation tables, for tests that build them without the migration."""
    return [Base.metadata.tables[name] for name in EVAL_TABLE_NAMES]

"""
app/models/record_access_log.py
RecordAccessLog model — one row each time an admin lists, views, downloads or
prints CTC records, so re-prints and look-ups can be traced to an account.
The table lives in supabase/migrations/20261018090000_records_audit.sql.
"""
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func

from app.core.database import Base

RECORD_ACCESS_ACTIONS = ("list", "view", "download", "print")


class RecordAccessLog(Base):
    __tablename__ = "record_access_log"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)

    staff_user_id = Column(Integer, ForeignKey("staff_users.id", ondelete="SET NULL"), nullable=True)
    action = Column(String(20), nullable=False)  # one of RECORD_ACCESS_ACTIONS

    # view / download / print: the record touched. ctc_number is a copy so the
    # row still reads well after the record itself is deleted.
    issuance_id = Column(Integer, ForeignKey("ctc_issuance.id", ondelete="SET NULL"), nullable=True)
    ctc_number = Column(String(30), nullable=True)
    is_copy = Column(Boolean, nullable=True)  # download / print: True when it got the COPY mark

    search_text = Column(Text, nullable=True)  # list: what the admin searched for

    staff = relationship("StaffUser")

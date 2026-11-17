"""
tests/test_admin_seed.py
Plan 010: the first admin account comes only from ADMIN_INITIAL_PASSWORD; no password is in the code or the log.
Each test seeds a throwaway SQLite database of its own (never the shared test DB or Supabase).
Run with: pytest tests/ -v
"""
# ruff: noqa: F811, I001  (fixtures from test_backend are re-used as arguments; it must be imported first)
import re
from pathlib import Path
from unittest.mock import patch

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from tests.test_backend import client, setup_test_db  # noqa: F401  (sets the throwaway env first)
import app.main as main_module
from app.core.database import Base, get_db
from app.core.security import hash_password, verify_password
from app.models.staff_user import RoleEnum, StaffUser

FAKE_PASSWORD = "Fake-Seed-Pass-2026"   # fake test value, 19 characters
SHORT_PASSWORD = "fake-short1"          # fake test value, 11 characters (1 under the minimum)
OLD_STARTER = "admin" + "1234"          # the removed starter password, built so no scan finds it as a literal
APP_DIR = Path(__file__).resolve().parent.parent / "app"


@pytest.fixture()
def seed_db(tmp_path):
    """A throwaway SQLite database with empty tables; _seed_default_admin() writes to it."""
    engine = create_engine(f"sqlite:///{tmp_path / 'seed.db'}", connect_args={"check_same_thread": False})
    Base.metadata.create_all(bind=engine)
    session_factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    with patch("app.core.database.SessionLocal", session_factory):
        yield session_factory
    engine.dispose()


def _seed(caplog, password):
    caplog.set_level("INFO")
    with patch.object(main_module.settings, "ADMIN_INITIAL_PASSWORD", password):
        main_module._seed_default_admin()


def _staff(session_factory):
    db = session_factory()
    try:
        return db.query(StaffUser).all()
    finally:
        db.close()


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


class TestAdminSeed:
    def test_empty_database_without_the_setting_creates_no_admin(self, seed_db, caplog):
        _seed(caplog, "")
        assert _staff(seed_db) == []
        warnings = _warnings(caplog)
        assert len(warnings) == 1
        assert "no admin account was created" in warnings[0] and "ADMIN_INITIAL_PASSWORD" in warnings[0]

    @pytest.mark.parametrize("value", [SHORT_PASSWORD, " " * 20])
    def test_short_or_blank_setting_creates_no_admin(self, seed_db, caplog, value):
        _seed(caplog, value)
        assert _staff(seed_db) == []
        assert len(_warnings(caplog)) == 1
        if value.strip():
            assert not [r for r in caplog.records if value in r.getMessage()]

    def test_valid_setting_creates_one_admin_that_can_log_in(self, seed_db, caplog, client):
        _seed(caplog, FAKE_PASSWORD)
        staff = _staff(seed_db)
        assert [(u.username, u.role, u.position) for u in staff] == [("admin", RoleEnum.ADMIN, "IT Administrator")]
        assert staff[0].hashed_password != FAKE_PASSWORD
        assert _warnings(caplog) == [(
            "First admin account 'admin' created from ADMIN_INITIAL_PASSWORD. "
            "Change it after the first login and remove the variable."
        )]

        def seed_session():
            db = seed_db()
            try:
                yield db
            finally:
                db.close()

        saved = main_module.app.dependency_overrides.get(get_db)
        main_module.app.dependency_overrides[get_db] = seed_session
        try:
            ok = client.post("/api/auth/login", json={"username": "admin", "password": FAKE_PASSWORD})
            old = client.post("/api/auth/login", json={"username": "admin", "password": OLD_STARTER})
        finally:
            main_module.app.dependency_overrides[get_db] = saved
        assert ok.status_code == 200 and ok.json()["access_token"]
        assert old.status_code == 401
        assert FAKE_PASSWORD not in ok.text

    def test_password_is_never_logged(self, seed_db, caplog):
        _seed(caplog, FAKE_PASSWORD)
        assert caplog.records
        assert not [r for r in caplog.records if FAKE_PASSWORD in r.getMessage()]

    def test_existing_staff_are_left_alone(self, seed_db, caplog):
        db = seed_db()
        db.add(StaffUser(username="fake_clerk", hashed_password=hash_password("fake-clerk-pass"),
                         full_name="Fake Clerk", role=RoleEnum.STAFF))
        db.commit()
        db.close()

        _seed(caplog, FAKE_PASSWORD)
        staff = _staff(seed_db)
        assert [u.username for u in staff] == ["fake_clerk"]
        assert verify_password("fake-clerk-pass", staff[0].hashed_password)
        assert _warnings(caplog) == []


class TestNoPasswordInSource:
    def test_no_hash_password_call_with_a_string_literal(self):
        call = re.compile(r"""hash_password\(\s*[rRbBuUfF]*["']""")
        found = [f"{path.relative_to(APP_DIR)}:{n}"
                 for path in APP_DIR.rglob("*.py")
                 for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
                 if call.search(line)]
        assert found == []

    def test_old_starter_password_is_gone_from_app(self):
        found = [str(path.relative_to(APP_DIR)) for path in APP_DIR.rglob("*.py")
                 if OLD_STARTER in path.read_text(encoding="utf-8")]
        assert found == []

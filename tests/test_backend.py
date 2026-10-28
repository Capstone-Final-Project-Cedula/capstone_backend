"""
tests/test_backend.py
Comprehensive unit and integration tests for the Cedula System backend.
Run with: pytest tests/ -v
"""
import pytest
import json
import os
from datetime import date, datetime, timedelta
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# ─── Test Database Setup (in-memory SQLite) ────────────────────────────────────

TEST_DATABASE_URL = "sqlite:///./test_cedula.db"

os.environ["DATABASE_URL"] = TEST_DATABASE_URL
os.environ["JWT_SECRET_KEY"] = "test-jwt-secret-key-32chars-minimum"
os.environ["SECRET_KEY"] = "test-secret-key-32chars-minimum-xx"
# Keep tests off the real Supabase Storage even when .env has credentials.
os.environ["SUPABASE_URL"] = ""
os.environ["SUPABASE_SECRET_KEY"] = ""
os.environ["NER_ENABLED"] = "false"   # don't load DistilBERT in unit tests
os.environ["WHISPER_PRELOAD"] = "false"
os.environ["WHISPER_BEAM_SIZE"] = "1"          # pinned: a local settings file may choose another beam, engine or table
os.environ["WHISPER_ENGINE"] = "openai"
os.environ["WHISPER_MODEL"] = "base"
os.environ["WHISPER_CONFIRM_MODEL"] = "tiny"
os.environ["WHISPER_FIELD_MODELS"] = "true"
os.environ["BARANGAY_NAME"] = "Barangay Imbatug"
os.environ["MUNICIPALITY"] = "Baungon"
os.environ["PROVINCE"] = "Bukidnon"

from app.main import app
from app.core.database import Base, get_db
from app.core.security import hash_password, create_access_token

engine = create_engine(TEST_DATABASE_URL, connect_args={"check_same_thread": False})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db


@pytest.fixture(scope="module", autouse=True)
def setup_test_db():
    """Create tables before tests, drop after."""
    from app.models import ApplicantRecord, CTCIssuance, MobileSubmission, StaffUser
    Base.metadata.create_all(bind=engine)
    yield
    Base.metadata.drop_all(bind=engine)
    # Windows cannot delete a SQLite file that still has open connections.
    from app.core.database import engine as app_engine
    engine.dispose()
    app_engine.dispose()
    if os.path.exists("test_cedula.db"):
        os.remove("test_cedula.db")


@pytest.fixture(scope="module")
def client():
    return TestClient(app)


@pytest.fixture(scope="module")
def db():
    db = TestingSessionLocal()
    yield db
    db.close()


@pytest.fixture(scope="module")
def admin_user(db):
    """Create and return a test admin user."""
    from app.models.staff_user import StaffUser, RoleEnum
    user = StaffUser(
        username="test_admin",
        hashed_password=hash_password("testpass123"),
        full_name="Test Admin",
        role=RoleEnum.ADMIN,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture(scope="module")
def staff_user(db):
    """Create and return a test staff user."""
    from app.models.staff_user import StaffUser, RoleEnum
    user = StaffUser(
        username="test_staff",
        hashed_password=hash_password("testpass123"),
        full_name="Test Staff",
        role=RoleEnum.STAFF,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


@pytest.fixture(scope="module")
def admin_token(admin_user):
    return create_access_token(data={"sub": admin_user.username, "type": "staff"})


@pytest.fixture(scope="module")
def staff_token(staff_user):
    return create_access_token(data={"sub": staff_user.username, "type": "staff"})


# ─── Auth Tests ─────────────────────────────────────────────────────────────────

class TestAuth:
    def test_login_success(self, client, admin_user):
        resp = client.post("/api/auth/login", json={
            "username": "test_admin",
            "password": "testpass123",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "access_token" in data
        assert data["token_type"] == "bearer"

    def test_login_wrong_password(self, client):
        resp = client.post("/api/auth/login", json={
            "username": "test_admin",
            "password": "wrongpassword",
        })
        assert resp.status_code == 401

    def test_login_nonexistent_user(self, client):
        resp = client.post("/api/auth/login", json={
            "username": "nobody",
            "password": "whatever",
        })
        assert resp.status_code == 401

    def test_get_me(self, client, admin_token):
        resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["username"] == "test_admin"
        assert data["role"] == "admin"

    def test_register_staff_as_admin(self, client, admin_token):
        resp = client.post(
            "/api/auth/register",
            json={
                "username": "new_staff",
                "password": "newpass123",
                "full_name": "New Staff Member",
                "role": "staff",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 201

    def test_register_duplicate_username(self, client, admin_token):
        resp = client.post(
            "/api/auth/register",
            json={
                "username": "test_admin",  # already exists
                "password": "pass12345",
                "full_name": "Duplicate",
                "role": "staff",
            },
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 409

    def test_register_requires_admin(self, client, staff_token):
        resp = client.post(
            "/api/auth/register",
            json={"username": "hack", "password": "hackpass123", "full_name": "Hack"},
            headers={"Authorization": f"Bearer {staff_token}"},
        )
        assert resp.status_code == 403


# ─── Tax Computation Tests ─────────────────────────────────────────────────────

class TestTaxService:
    def test_basic_tax_only(self):
        from app.services.tax_service import compute_tax
        result = compute_tax(gross_annual_income=0.0)
        assert result["basic_community_tax"] == 5.00
        assert result["additional_community_tax"] == 0.00
        assert result["total_tax"] == 5.00

    def test_additional_tax_calculation(self):
        from app.services.tax_service import compute_tax
        # ₱50,000 income → ₱50 additional tax
        result = compute_tax(gross_annual_income=50_000.0)
        assert result["additional_community_tax"] == 50.00
        assert result["total_tax"] == 55.00

    def test_additional_tax_cap(self):
        from app.services.tax_service import compute_tax
        # ₱10,000,000 income → capped at ₱5,000
        result = compute_tax(gross_annual_income=10_000_000.0)
        assert result["additional_community_tax"] == 5_000.00
        assert result["total_tax"] == 5_005.00

    def test_tax_with_property(self):
        from app.services.tax_service import compute_tax
        # Property value higher than income → use property value
        result = compute_tax(gross_annual_income=10_000.0, real_property_value=100_000.0)
        assert result["additional_community_tax"] == 100.00

    def test_tax_classification_with_income(self):
        from app.services.tax_service import compute_tax
        from app.models.applicant import TaxClassificationEnum
        result = compute_tax(gross_annual_income=50_000.0)
        assert result["tax_classification"] == TaxClassificationEnum.A

    def test_tax_classification_no_income(self):
        from app.services.tax_service import compute_tax
        from app.models.applicant import TaxClassificationEnum
        result = compute_tax(gross_annual_income=0.0)
        assert result["tax_classification"] == TaxClassificationEnum.C

    def test_tax_api_endpoint(self, client, staff_token):
        resp = client.post(
            "/api/extraction/tax?gross_annual_income=50000",
            headers={"Authorization": f"Bearer {staff_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_tax"] == 55.00


# ─── Regex Extraction Tests ────────────────────────────────────────────────────

class TestExtraction:
    def test_extract_height_cm(self):
        from app.services.extraction_service import _extract_height
        assert _extract_height("My height is 167 cm") == 167.0
        assert _extract_height("I am 170 centimeters tall") == 170.0

    def test_extract_height_feet(self):
        from app.services.extraction_service import _extract_height
        result = _extract_height("I am 5 feet 7 inches tall")
        assert result is not None
        assert abs(result - 170.18) < 0.5  # Roughly 170cm

    def test_extract_weight_kg(self):
        from app.services.extraction_service import _extract_weight
        assert _extract_weight("I weigh 65 kilograms") == 65.0
        assert _extract_weight("My weight is 70 kg") == 70.0

    def test_extract_weight_lbs(self):
        from app.services.extraction_service import _extract_weight
        result = _extract_weight("I weigh 143 pounds")
        assert result is not None
        assert abs(result - 64.86) < 0.5

    def test_extract_income_peso(self):
        from app.services.extraction_service import _extract_income
        assert _extract_income("My income is 50000 pesos") == 50000.0
        assert _extract_income("I earn ₱120,000 annually") == 120000.0

    def test_extract_civil_status(self):
        from app.services.extraction_service import _extract_civil_status
        assert _extract_civil_status("I am single") == "Single"
        assert _extract_civil_status("I am married") == "Married"
        assert _extract_civil_status("I am widowed") == "Widowed"

    def test_extract_sex(self):
        from app.services.extraction_service import _extract_sex
        assert _extract_sex("I am male") == "Male"
        assert _extract_sex("I am female") == "Female"

    def test_extract_date(self):
        from app.services.extraction_service import _extract_date
        result = _extract_date("I was born on January 15 1990")
        assert result == date(1990, 1, 15)

    def test_extract_occupation(self):
        from app.services.extraction_service import _extract_occupation
        result = _extract_occupation("I am working as a teacher")
        assert result is not None
        assert "teacher" in result.lower()

    def test_full_extraction_pipeline(self, client, staff_token):
        """Integration test: extract fields from a sample transcript."""
        transcript = (
            "My name is Juan dela Cruz. I was born on March 10 1985 in Baungon Bukidnon. "
            "I am male and single. I am a Filipino citizen. "
            "I am currently working as a teacher. "
            "My gross annual income is 240000 pesos. "
            "I live at Barangay Imbatug, Baungon, Bukidnon. "
            "My height is 165 centimeters and my weight is 60 kilograms."
        )
        resp = client.post(
            "/api/extraction/fields",
            json={"transcript": transcript},
            headers={"Authorization": f"Bearer {staff_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        fields = data["extracted_fields"]
        assert fields["gross_annual_income"] == 240000.0
        assert fields["civil_status"] == "Single"
        assert fields["sex"] == "Male"
        assert fields["height_cm"] == 165.0
        assert fields["weight_kg"] == 60.0


# ─── QR Code Tests ─────────────────────────────────────────────────────────────

class TestQRCode:
    SAMPLE_FORM = {
        "surname": "Dela Cruz",
        "given_name": "Juan",
        "middle_name": "Santos",
        "date_of_birth": "1990-01-15",
        "address": "Barangay Imbatug, Baungon, Bukidnon",
        "sex": "Male",
        "civil_status": "Single",
        "citizenship": "Filipino",
        "icr_number": None,
        "place_of_birth": "Baungon",
        "height_cm": 165.0,
        "weight_kg": 60.0,
        "occupation": "Teacher",
        "gross_annual_income": 240000.0,
        "tax_classification": "A",
    }

    def test_generate_qr(self, client):
        resp = client.post("/api/qr/generate", json={"form_data": self.SAMPLE_FORM})
        assert resp.status_code == 201
        data = resp.json()
        assert "qr_token" in data
        assert "qr_image_base64" in data
        assert "expires_at" in data
        assert len(data["qr_token"]) == 32
        self.__class__.last_token = data["qr_token"]

    def test_status_before_scan(self, client):
        resp = client.get(f"/api/qr/status/{self.__class__.last_token}")
        assert resp.status_code == 200
        assert resp.json() == {"is_claimed": False, "is_expired": False}

    def test_scan_qr_valid_token(self, client, staff_token):
        token = self.__class__.last_token
        resp = client.post(
            "/api/qr/scan",
            json={"qr_token": token},
            headers={"Authorization": f"Bearer {staff_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["form_data"]["surname"] == "Dela Cruz"
        assert data["form_data"]["given_name"] == "Juan"

    def test_status_after_scan(self, client):
        resp = client.get(f"/api/qr/status/{self.__class__.last_token}")
        assert resp.json() == {"is_claimed": True, "is_expired": False}
        assert client.get("/api/qr/status/not-a-real-token").status_code == 404

    def test_scan_qr_already_used(self, client, staff_token):
        """Token should be invalidated after first use."""
        token = self.__class__.last_token
        resp = client.post(
            "/api/qr/scan",
            json={"qr_token": token},
            headers={"Authorization": f"Bearer {staff_token}"},
        )
        assert resp.status_code == 404

    def test_scan_qr_invalid_token(self, client, staff_token):
        resp = client.post(
            "/api/qr/scan",
            json={"qr_token": "INVALIDTOKEN000000000000000000000"},
            headers={"Authorization": f"Bearer {staff_token}"},
        )
        assert resp.status_code == 404


# ─── CTC Issuance Tests ────────────────────────────────────────────────────────

class TestIssuance:
    SAMPLE_APPLICANT = {
        "surname": "Reyes",
        "given_name": "Maria",
        "middle_name": "Santos",
        "date_of_birth": "1995-06-20",
        "address": "Barangay Imbatug, Baungon, Bukidnon",
        "sex": "Female",
        "civil_status": "Single",
        "citizenship": "Filipino",
        "icr_number": None,
        "place_of_birth": "Bukidnon",
        "height_cm": 158.0,
        "weight_kg": 52.0,
        "occupation": "Government Employee",
        "gross_annual_income": 360000.0,
        "tax_classification": "A",
    }

    def test_issue_ctc(self, client, staff_token):
        with patch("app.api.routes.forms.generate_cedula_pdf", return_value="pdfs/test.pdf"), \
             patch("app.api.routes.forms.generate_receipt_pdf", return_value="pdfs/test_receipt.pdf"):
            resp = client.post(
                "/api/forms/issue",
                json={
                    "applicant_data": self.SAMPLE_APPLICANT,
                    "input_method": "voice",
                    "raw_transcript": "Test transcript",
                },
                headers={"Authorization": f"Bearer {staff_token}"},
            )
        assert resp.status_code == 201
        data = resp.json()
        assert "ctc_number" in data
        assert data["ctc_number"].startswith("CTC-")
        assert data["total_tax"] == 365.00  # 5 + 360
        self.__class__.last_ctc = data["ctc_number"]
        self.__class__.last_id = data["id"]

    def test_get_record(self, client, admin_token):
        resp = client.get(
            f"/api/forms/records/{self.__class__.last_id}",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ctc_number"] == self.__class__.last_ctc

    def test_list_records(self, client, admin_token):
        resp = client.get(
            "/api/forms/records",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) >= 1

    def test_search_records(self, client, admin_token):
        resp = client.get(
            "/api/forms/records?search=Reyes",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        records = resp.json()
        assert any("Reyes" in r["applicant"]["surname"] for r in records)

    def test_dashboard_stats(self, client, admin_token):
        resp = client.get(
            "/api/forms/dashboard/stats",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_issued"] >= 1
        assert data["today_issued"] >= 1
        assert len(data["daily_data"]) == 7

    def test_dashboard_is_admin_only(self, client, staff_token):
        resp = client.get(
            "/api/forms/dashboard/stats",
            headers={"Authorization": f"Bearer {staff_token}"},
        )
        assert resp.status_code == 403


# ─── Plan 001.3: required fields, admin-only Records, audit log, COPY mark ─────

def _auth(token):
    return {"Authorization": f"Bearer {token}"}


def _issue(client, token, pdf_mocked=True, **changes):
    """POST /api/forms/issue with SAMPLE_APPLICANT plus `changes` (fake data only)."""
    applicant = {**TestIssuance.SAMPLE_APPLICANT, **changes}
    body = {"applicant_data": applicant, "input_method": "voice"}
    if not pdf_mocked:
        return client.post("/api/forms/issue", json=body, headers=_auth(token))
    with patch("app.api.routes.forms.generate_cedula_pdf", return_value="pdfs/test.pdf"), \
         patch("app.api.routes.forms.generate_receipt_pdf", return_value="pdfs/test_receipt.pdf"):
        return client.post("/api/forms/issue", json=body, headers=_auth(token))


def _row_counts():
    from app.models import ApplicantRecord, CTCIssuance
    session = TestingSessionLocal()
    try:
        return session.query(ApplicantRecord).count(), session.query(CTCIssuance).count()
    finally:
        session.close()


def _pdf_text(content: bytes) -> str:
    """Text drawn on a ReportLab PDF (its streams are ASCII85 + Flate encoded)."""
    import base64
    import re
    import zlib
    parts = []
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", content, re.S):
        raw = match.group(1).strip()
        raw = raw[:-2] if raw.endswith(b"~>") else raw
        try:
            parts.append(zlib.decompress(base64.a85decode(raw)).decode("latin-1"))
        except Exception:
            continue
    return "\n".join(parts)


def _print_count(ctc_number: str) -> int:
    from app.models import CTCIssuance
    session = TestingSessionLocal()
    try:
        return session.query(CTCIssuance).filter_by(ctc_number=ctc_number).one().print_count
    finally:
        session.close()


def _access_log(client, admin_token, limit=100):
    resp = client.get(f"/api/forms/access-log?limit={limit}", headers=_auth(admin_token))
    assert resp.status_code == 200
    return resp.json()


class TestIssueRequiredFields:
    @pytest.mark.parametrize("field,value", [
        ("surname", ""),
        ("given_name", "   "),
        ("address", ""),
        ("sex", None),
        ("sex", "unknown"),
        ("civil_status", ""),
        ("date_of_birth", None),
        ("date_of_birth", "not a date"),
    ])
    def test_blank_required_field_is_422_and_writes_nothing(self, client, staff_token, field, value):
        before = _row_counts()
        with patch("app.api.routes.forms.generate_cedula_pdf") as make_pdf, \
             patch("app.api.routes.forms.generate_receipt_pdf") as make_receipt:
            resp = client.post(
                "/api/forms/issue",
                json={"applicant_data": {**TestIssuance.SAMPLE_APPLICANT, field: value}, "input_method": "voice"},
                headers=_auth(staff_token),
            )
        assert resp.status_code == 422
        assert field in [item["loc"][-1] for item in resp.json()["detail"]]
        assert _row_counts() == before
        make_pdf.assert_not_called()
        make_receipt.assert_not_called()

    def test_several_missing_fields_are_all_listed(self, client, staff_token):
        resp = _issue(client, staff_token, surname="", address="", civil_status="")
        assert resp.status_code == 422
        assert {item["loc"][-1] for item in resp.json()["detail"]} == {"surname", "address", "civil_status"}

    def test_blank_height_and_weight_still_issue(self, client, staff_token):
        # The web sends 0 for a blank height/weight; that must keep working.
        resp = _issue(client, staff_token, height_cm=0, weight_kg=0)
        assert resp.status_code == 201
        assert resp.json()["applicant"]["height_cm"] is None

    def test_complete_application_is_201(self, client, staff_token):
        assert _issue(client, staff_token).status_code == 201


class TestRecordsAdminOnly:
    def test_staff_gets_403_on_list_view_download(self, client, staff_token, admin_token):
        record = _issue(client, staff_token).json()
        assert client.get("/api/forms/records", headers=_auth(staff_token)).status_code == 403
        assert client.get(f"/api/forms/records/{record['id']}", headers=_auth(staff_token)).status_code == 403
        assert client.get(f"/api/forms/download/{record['ctc_number']}",
                          headers=_auth(staff_token)).status_code == 403
        # Admin can list and view (download is covered with a real PDF below).
        assert client.get("/api/forms/records", headers=_auth(admin_token)).status_code == 200
        assert client.get(f"/api/forms/records/{record['id']}", headers=_auth(admin_token)).status_code == 200

    def test_staff_gets_403_on_access_log(self, client, staff_token):
        assert client.get("/api/forms/access-log", headers=_auth(staff_token)).status_code == 403


class TestRecordsNameSearch:
    @pytest.fixture(scope="class")
    def people(self, client, staff_token):
        juan = _issue(client, staff_token, surname="Dela Cruz", given_name="Juan", middle_name="Bautista").json()
        pedro = _issue(client, staff_token, surname="Dela Cruz", given_name="Pedro", middle_name="Lim").json()
        concepcion = _issue(client, staff_token, surname="Dela Cruz-Villanueva",
                            given_name="Ma. Concepcion", middle_name=None).json()
        return {"juan": juan["id"], "pedro": pedro["id"], "concepcion": concepcion["id"]}

    def _found(self, client, admin_token, search):
        resp = client.get("/api/forms/records", params={"search": search, "limit": 100},
                          headers=_auth(admin_token))
        assert resp.status_code == 200
        return {row["id"] for row in resp.json()}

    @pytest.mark.parametrize("search", ["Juan Dela Cruz", "Dela Cruz Juan", "juan", "  JUAN   dela  ", "Bautista Juan"])
    def test_word_search_finds_juan(self, client, admin_token, people, search):
        assert people["juan"] in self._found(client, admin_token, search)

    def test_other_given_name_does_not_match(self, client, admin_token, people):
        found = self._found(client, admin_token, "Juan Dela Cruz")
        assert people["pedro"] not in found
        assert people["juan"] not in self._found(client, admin_token, "Pedro Dela Cruz")

    def test_long_name_found_by_two_words(self, client, admin_token, people):
        assert people["concepcion"] in self._found(client, admin_token, "Concepcion Villanueva")

    def test_ctc_number_search_still_works(self, client, admin_token, people):
        record = client.get(f"/api/forms/records/{people['pedro']}", headers=_auth(admin_token)).json()
        assert self._found(client, admin_token, record["ctc_number"]) == {people["pedro"]}

    def test_like_wildcards_are_literal(self, client, admin_token, people):
        assert self._found(client, admin_token, "%") == set()
        assert self._found(client, admin_token, "J_an") == set()


class TestRecordsAccessLogAndCopies:
    def test_view_and_list_are_logged(self, client, staff_token, admin_token):
        record = _issue(client, staff_token).json()
        client.get(f"/api/forms/records/{record['id']}", headers=_auth(admin_token))
        client.get("/api/forms/records", params={"search": "Reyes Maria"}, headers=_auth(admin_token))
        rows = _access_log(client, admin_token)
        assert any(r["action"] == "view" and r["issuance_id"] == record["id"]
                   and r["ctc_number"] == record["ctc_number"] and r["staff_username"] == "test_admin"
                   for r in rows)
        assert any(r["action"] == "list" and r["search_text"] == "Reyes Maria" for r in rows)

    def test_list_without_search_is_not_logged(self, client, admin_token):
        before = len(_access_log(client, admin_token))
        client.get("/api/forms/records?skip=10", headers=_auth(admin_token))
        client.get("/api/forms/records?search=%20%20", headers=_auth(admin_token))
        assert len(_access_log(client, admin_token)) == before

    def test_first_print_is_original_then_copies(self, client, staff_token, admin_token, tmp_path, monkeypatch):
        from app.core.config import settings
        monkeypatch.setattr(settings, "PDF_OUTPUT_DIR", str(tmp_path))

        record = _issue(client, staff_token, pdf_mocked=False).json()
        ctc = record["ctc_number"]
        stored = open(record["pdf_path"], "rb").read()

        first = client.get(f"/api/forms/download/{ctc}?purpose=print", headers=_auth(admin_token))
        assert first.status_code == 200
        assert first.content == stored
        assert "COPY / RE-PRINT" not in _pdf_text(first.content)
        assert "_COPY" not in first.headers["content-disposition"]

        second = client.get(f"/api/forms/download/{ctc}?purpose=print", headers=_auth(admin_token))
        assert second.status_code == 200
        assert "COPY / RE-PRINT" in _pdf_text(second.content)
        assert f"{ctc}_COPY.pdf" in second.headers["content-disposition"]

        # The stored original is never changed, and no temporary re-print is left behind.
        assert open(record["pdf_path"], "rb").read() == stored
        assert not [name for name in os.listdir(tmp_path) if "_copy_" in name]
        assert _print_count(ctc) == 2

        mine = [r for r in _access_log(client, admin_token) if r["ctc_number"] == ctc]
        assert sorted((r["action"], r["is_copy"]) for r in mine) == [("print", False), ("print", True)]

    def test_download_is_always_a_copy_and_never_uses_up_the_original(
        self, client, staff_token, admin_token, tmp_path, monkeypatch,
    ):
        from app.core.config import settings
        monkeypatch.setattr(settings, "PDF_OUTPUT_DIR", str(tmp_path))

        record = _issue(client, staff_token, pdf_mocked=False).json()
        ctc = record["ctc_number"]
        stored = open(record["pdf_path"], "rb").read()

        for _ in range(2):
            download = client.get(f"/api/forms/download/{ctc}?purpose=download", headers=_auth(admin_token))
            assert download.status_code == 200
            assert "COPY / RE-PRINT" in _pdf_text(download.content)
            assert f"{ctc}_COPY.pdf" in download.headers["content-disposition"]
        # The default purpose is a download too.
        assert "COPY / RE-PRINT" in _pdf_text(
            client.get(f"/api/forms/download/{ctc}", headers=_auth(admin_token)).content)
        assert _print_count(ctc) == 0

        # The first Print after any number of Downloads is still the original.
        printed = client.get(f"/api/forms/download/{ctc}?purpose=print", headers=_auth(admin_token))
        assert printed.content == stored
        assert "_COPY" not in printed.headers["content-disposition"]
        assert _print_count(ctc) == 1

        mine = [r for r in _access_log(client, admin_token) if r["ctc_number"] == ctc]
        assert sorted((r["action"], r["is_copy"]) for r in mine) == [
            ("download", True), ("download", True), ("download", True), ("print", False),
        ]

    def test_failed_download_does_not_use_up_the_original(self, client, staff_token, admin_token):
        from app.models import CTCIssuance
        record = _issue(client, staff_token).json()   # mocked PDF path: the file does not exist
        resp = client.get(f"/api/forms/download/{record['ctc_number']}?purpose=print",
                          headers=_auth(admin_token))
        assert resp.status_code == 404
        session = TestingSessionLocal()
        try:
            assert session.query(CTCIssuance).filter_by(id=record["id"]).one().print_count == 0
        finally:
            session.close()

    def test_download_works_when_the_log_write_fails(self, client, staff_token, admin_token, tmp_path, monkeypatch):
        from app.core.config import settings
        monkeypatch.setattr(settings, "PDF_OUTPUT_DIR", str(tmp_path))
        record = _issue(client, staff_token, pdf_mocked=False).json()
        with patch("app.api.routes.forms.RecordAccessLog", side_effect=RuntimeError("log table missing")):
            resp = client.get(f"/api/forms/download/{record['ctc_number']}", headers=_auth(admin_token))
        assert resp.status_code == 200
        assert resp.content.startswith(b"%PDF")

    def test_unknown_purpose_is_rejected(self, client, admin_token):
        assert client.get("/api/forms/download/CTC-X?purpose=email",
                          headers=_auth(admin_token)).status_code == 422


# ─── Plan 001.4: start-up migration check, QR with a blank birth date ─────────

class TestStartupMigrationCheck:
    @staticmethod
    def _engine(tmp_path, *statements):
        from sqlalchemy import text
        throwaway = create_engine(f"sqlite:///{tmp_path / 'schema.db'}")
        with throwaway.begin() as conn:
            for statement in statements:
                conn.execute(text(statement))
        return throwaway

    def test_empty_database_passes(self, tmp_path):
        from app.core.database import missing_migration_message
        throwaway = self._engine(tmp_path)
        assert missing_migration_message(throwaway) is None
        throwaway.dispose()

    def test_migrated_database_passes(self):
        # The test database is built from the models, which include the migration.
        from app.core.database import missing_migration_message
        assert missing_migration_message(engine) is None

    @pytest.mark.parametrize("statements", [
        # Before the migration: no print_count, no record_access_log.
        ["CREATE TABLE ctc_issuance (id INTEGER PRIMARY KEY)"],
        # Half-applied: the column is there but the table is not.
        ["CREATE TABLE ctc_issuance (id INTEGER PRIMARY KEY, print_count INTEGER)"],
        # The table is there (e.g. made by create_all) but the column is not.
        ["CREATE TABLE ctc_issuance (id INTEGER PRIMARY KEY)",
         "CREATE TABLE record_access_log (id INTEGER PRIMARY KEY)"],
    ])
    def test_unmigrated_database_is_named(self, tmp_path, statements):
        from app.core.database import missing_migration_message
        throwaway = self._engine(tmp_path, *statements)
        message = missing_migration_message(throwaway)
        throwaway.dispose()
        assert message and "20261018090000_records_audit.sql" in message

    def test_backend_refuses_to_start_without_the_migration(self, tmp_path, caplog):
        import app.main as main_module
        caplog.set_level("INFO")
        throwaway = self._engine(tmp_path, "CREATE TABLE ctc_issuance (id INTEGER PRIMARY KEY)")
        with patch("app.core.database.engine", throwaway), \
             patch.object(main_module, "create_tables") as make_tables, \
             patch.object(main_module, "_seed_default_admin") as seed:
            with pytest.raises(RuntimeError, match="20261018090000_records_audit.sql"):
                with TestClient(app):
                    pass
        throwaway.dispose()
        # It stopped before create_all could make record_access_log without RLS.
        make_tables.assert_not_called()
        seed.assert_not_called()
        errors = [r for r in caplog.records if r.levelname == "ERROR" and "records_audit" in r.getMessage()]
        assert len(errors) == 1
        assert any(r.getMessage() == "Database: SQLite (test only)" for r in caplog.records)

    def test_backend_starts_with_the_migration(self, caplog):
        import app.main as main_module
        caplog.set_level("INFO")
        with patch.object(main_module, "_seed_default_admin"):
            with TestClient(app) as started:
                assert started.get("/api/health").status_code == 200
        assert not [r for r in caplog.records if r.levelname == "ERROR"]
        # The URL (and its password) is never logged.
        assert not [r for r in caplog.records if "sqlite:///" in r.getMessage()]


class TestQRBlankBirthDate:
    def test_scan_then_issue_without_birth_date(self, client, staff_token):
        form = {**TestQRCode.SAMPLE_FORM, "date_of_birth": None}
        token = client.post("/api/qr/generate", json={"form_data": form}).json()["qr_token"]

        scanned = client.post("/api/qr/scan", json={"qr_token": token}, headers=_auth(staff_token))
        assert scanned.status_code == 200
        assert scanned.json()["form_data"]["date_of_birth"] is None

        before = _row_counts()
        with patch("app.api.routes.forms.generate_cedula_pdf") as make_pdf, \
             patch("app.api.routes.forms.generate_receipt_pdf") as make_receipt:
            resp = client.post(
                "/api/forms/issue",
                json={"applicant_data": scanned.json()["form_data"], "input_method": "qr"},
                headers=_auth(staff_token),
            )
        assert resp.status_code == 422
        assert [item["loc"][-1] for item in resp.json()["detail"]] == ["date_of_birth"]
        assert _row_counts() == before
        make_pdf.assert_not_called()
        make_receipt.assert_not_called()


# ─── Permission / Token Tests ──────────────────────────────────────────────────

class TestPermissions:
    def test_citizen_token_rejected_by_staff_routes(self, client, admin_user):
        # A citizen whose username equals a staff username must not get staff access.
        citizen_token = create_access_token(data={"sub": admin_user.username, "type": "citizen"})
        resp = client.get("/api/auth/staff", headers={"Authorization": f"Bearer {citizen_token}"})
        assert resp.status_code == 401

    def test_untyped_token_rejected(self, client, admin_user):
        legacy = create_access_token(data={"sub": admin_user.username})
        resp = client.get("/api/auth/me", headers={"Authorization": f"Bearer {legacy}"})
        assert resp.status_code == 401

    def test_viewer_cannot_issue(self, client, db):
        from app.models.staff_user import StaffUser, RoleEnum
        viewer = StaffUser(username="test_viewer", hashed_password=hash_password("testpass123"),
                           full_name="Test Viewer", role=RoleEnum.VIEWER)
        db.add(viewer)
        db.commit()
        token = create_access_token(data={"sub": viewer.username, "type": "staff"})
        headers = {"Authorization": f"Bearer {token}"}
        assert client.get("/api/forms/records", headers=headers).status_code == 403  # Records is admin only
        resp = client.post("/api/forms/issue", headers=headers,
                           json={"applicant_data": TestIssuance.SAMPLE_APPLICANT, "input_method": "manual"})
        assert resp.status_code == 403

    def test_admin_cannot_deactivate_self(self, client, admin_user, admin_token):
        resp = client.patch(f"/api/auth/staff/{admin_user.id}/toggle-active",
                            headers={"Authorization": f"Bearer {admin_token}"})
        assert resp.status_code == 400

    def test_pdfs_not_publicly_served(self, client):
        assert client.get("/pdfs/CTC_20260101_0001.pdf").status_code == 404


# ─── Live interview answers ────────────────────────────────────────────────────

class TestInterviewAnswers:
    @pytest.mark.parametrize("key,spoken,value", [
        ("sex", "I'm female po", "Female"),
        ("sex", "lalaki", "Male"),
        ("civil_status", "not married", "Single"),
        ("civil_status", "biyuda", "Widowed"),
        ("date", "May 10 1990", "May 10, 1990"),
        ("date", "the tenth of May nineteen ninety", "May 10, 1990"),
        ("date", "ika-10 ng Mayo 1990", "May 10, 1990"),
        ("date", "05/10/1990", "May 10, 1990"),
        ("date", "25/12/1985", "December 25, 1985"),
        ("date", "March twenty first nineteen ninety five", "March 21, 1995"),
        ("height", "five feet six inches", "168"),
        ("height", "160", "160"),
        ("weight", "130 pounds", "59"),
        ("gross_annual_income", "25 thousand a month", "300000"),
        ("gross_annual_income", "two hundred fifty thousand pesos", "250000"),
        ("gross_annual_income", "wala po", "0"),
        ("citizenship", "Pilipino", "Filipino"),
        ("icr_no", "none", ""),
        ("icr_no", "zero one two three four", "01234"),
        ("surname", "my surname is dela cruz", "Dela Cruz"),
        ("surname", "C R U Z", "Cruz"),
        ("middle_name", "wala po", ""),
        ("occupation", "I'm a tricycle driver", "Tricycle Driver"),
    ])
    def test_understands_natural_answers(self, key, spoken, value):
        from app.services.answer_service import understand_answer
        result = understand_answer(key, spoken)
        assert result.ok, result.message
        assert result.value == value

    @pytest.mark.parametrize("key,spoken", [
        ("sex", "nails"), ("civil_status", "nails"), ("date", "nails"), ("date", "May 10"),
        ("date", "February 30 1990"), ("date", "May 10 2090"), ("height", "nails"), ("height", "900"),
        ("weight", "nails"), ("gross_annual_income", "nails"), ("citizenship", "nails"),
        ("address", "nails"), ("surname", "12345"), ("icr_no", "yes"), ("sex", "can you repeat that"),
    ])
    def test_rejects_answers_that_do_not_fit_the_question(self, key, spoken):
        from app.services.answer_service import understand_answer
        result = understand_answer(key, spoken)
        assert not result.ok
        assert result.message

    @pytest.mark.parametrize("spoken,value", [
        ("Purok 2 Imbatug", "Purok 2 Imbatug, Baungon, Bukidnon"),
        ("Purik 2, Baranay Imbatog", "Purok 2, Barangay Imbatug, Baungon, Bukidnon"),   # misheard by speech engine
        ("sa Salim Balan", "Barangay Salimbalan, Baungon, Bukidnon"),
        ("Poblacion, Baungon", "Barangay Imbatug, Baungon, Bukidnon"),                   # old name of Imbatug
        ("Purok 5, Lingatin", "Purok 5, Barangay Lingating, Baungon, Bukidnon"),
        # Real Whisper (base) outputs for the Windows voice saying these addresses:
        ("Purok, too. Barangay in Baedig.", "Purok 2, Barangay Imbatug, Baungon, Bukidnon"),
        ("Barana Isolim Balan.", "Barangay Salimbalan, Baungon, Bukidnon"),
        ("PURIC 2. Baranati imbedig.", "Purok 2, Barangay Imbatug, Baungon, Bukidnon"),
        ("Purok V, Lingating.", "Purok 5, Lingating, Baungon, Bukidnon"),
    ])
    def test_address_uses_baungon_gazetteer(self, spoken, value):
        from app.services.answer_service import understand_answer
        result = understand_answer("address", spoken)
        assert result.ok and not result.needs_confirmation, result
        assert result.value == value

    def test_unknown_barangay_asks_for_confirmation(self):
        from app.services.answer_service import understand_answer
        result = understand_answer("address", "Purok 3, Morana Ibalbegon")
        assert result.ok and result.needs_confirmation

    def test_phonetic_keys_match_sound_alikes(self):
        from app.services.answer_service import phonetic_key, understand_answer
        assert phonetic_key("Imbatug") == phonetic_key("in Baedig")
        assert phonetic_key("Salimbalan") == phonetic_key("Isolim Balan")
        assert understand_answer("given_name", "1.").value == "Juan"   # "Juan" heard as "one"

    @pytest.mark.parametrize("key,heard,value", [
        # Real Whisper (base) outputs for Philippine voices saying these names:
        ("surname", "And what's her name is Bakulio?", "Baculio"),   # misheard lead-in + English spelling
        ("surname", "A vice surname is Sinagin.", "Sinagin"),
        ("surname", "Abelana.", "Abellana"),                          # known local spelling
        ("surname", "Pakana Po.", "Pacana"),
        ("surname", "Da Gondon.", "Dagondon"),                        # split into English words
        ("surname", "I got po.", "Igot"),
        ("surname", "My surname is Juma Wan.", "Jumawan"),
        ("surname", "Juma 1.", "Jumawan"),                            # "wan" written as a digit
        ("surname", "Baculiope.", "Baculio"),                         # "po" glued on
        ("surname", "Santos, Bo.", "Santos"),                         # "po" heard as "Bo"
        ("surname", "Dela Cruz po", "Dela Cruz"),                     # particle: stays two words
        ("given_name", "Maria, Pa.", "Maria"),
        ("given_name", "Jhon Paul", "Jhon Paul"),                     # given names are not joined
        ("given_name", "Ross, John.", "Ross John"),                   # a pause written as a comma
        ("given_name", "Mary. Ann.", "Mary Ann"),
        ("given_name", "Ma. Cristina", "Ma. Cristina"),               # abbreviation kept
        ("given_name", "M, A, R, Y, space, A, N, N.", "Mary Ann"),
        ("given_name", "Maria Cristina Luz", "Maria Cristina Luz"),   # three given names
        ("surname", "Ocampo", "Ocampo"),                              # a real "-po" name is kept
    ])
    def test_spoken_names(self, key, heard, value):
        from app.services.answer_service import understand_answer
        result = understand_answer(key, heard, spoken=True)
        assert result.ok, result.message
        assert result.value == value

    @pytest.mark.parametrize("heard", ["It's.", "Or these.", "this is", "That is it."])
    def test_filler_is_not_a_name(self, heard):
        from app.services.answer_service import understand_answer
        result = understand_answer("surname", heard, spoken=True)
        assert not result.ok and result.reason == "not_a_name"

    def test_typed_names_are_kept_as_typed(self):
        from app.services.answer_service import understand_answer
        assert understand_answer("surname", "Bakos").value == "Bakos"            # not respelled
        assert understand_answer("surname", "Santos Reyes").value == "Santos Reyes"

    def test_name_sound_key(self):
        from app.services.name_lexicon import sound_key
        assert sound_key("Baculio") == sound_key("Bakulio")
        assert sound_key("Abellana") == sound_key("Abelana")
        assert sound_key("Maglinte") == sound_key("Maglenti")
        assert sound_key("Ortega") != sound_key("Ordega")

    def test_ner_span_keeps_whole_words(self):
        from app.services import answer_service
        from app.services.answer_service import Entity, _handle_name
        # DistilBERT tagging only "mampos" of "Tumampos" must not cut the name.
        with patch.object(answer_service, "_ner_entities", return_value=[Entity("PER", "mampos", 0.9, 2, 8)]):
            assert _handle_name("surname", "Tumampos", "surname", optional=False).value == "Tumampos"

    def test_whisper_drops_silence_hallucinations(self):
        from app.services.speech_service import _clean_transcript, _decode_options
        assert _clean_transcript(" Thank you. ") == ""
        assert _clean_transcript("Thanks for watching.") == ""
        assert _clean_transcript("Female.") == "Female."
        assert "beam_size" not in _decode_options(None)   # greedy by default (WHISPER_BEAM_SIZE=1)

    def test_unrecognized_place_or_job_asks_for_confirmation(self):
        from app.services.answer_service import understand_answer
        assert understand_answer("place_of_birth", "nails").needs_confirmation
        assert understand_answer("occupation", "nails").needs_confirmation
        assert not understand_answer("occupation", "farmer").needs_confirmation

    def test_answer_endpoint(self, client, staff_token):
        headers = {"Authorization": f"Bearer {staff_token}"}
        r = client.post("/api/extraction/answer", headers=headers, json={"field_key": "sex", "transcript": "nails"})
        assert r.status_code == 200 and r.json()["ok"] is False and r.json()["message"]
        r = client.post("/api/extraction/answer", headers=headers, json={"field_key": "date", "transcript": "May 10 1990"})
        assert r.json()["ok"] and r.json()["normalized"] == "1990-05-10"
        r = client.post("/api/extraction/answer", headers=headers, json={"field_key": "bogus", "transcript": "x"})
        assert r.status_code == 422
        assert client.post("/api/extraction/answer", json={"field_key": "sex", "transcript": "male"}).status_code == 401

    def test_answers_endpoint_checks_everything(self, client, staff_token):
        headers = {"Authorization": f"Bearer {staff_token}"}
        answers = {
            "surname": "Santos", "given_name": "Maria", "middle_name": "", "address": "Purok 2, Barangay Imbatug",
            "date": "May 10, 1990", "sex": "Female", "civil_status": "Single", "citizenship": "Filipino",
            "icr_no": "", "place_of_birth": "Baungon", "height": "160", "weight": "55",
            "occupation": "Teacher", "gross_annual_income": "240000",
        }
        r = client.post("/api/extraction/answers", headers=headers, json={"answers": answers})
        assert r.status_code == 200 and r.json()["ok"] is True
        r = client.post("/api/extraction/answers", headers=headers, json={"answers": {**answers, "sex": "nails"}})
        assert r.json()["ok"] is False and r.json()["results"]["sex"]["ok"] is False


# ─── Plan 001.6: the problems from the 10-20 interview, town + barangay matching ──

class TestSpeechAccuracy:
    @pytest.mark.parametrize("heard", [
        "no", "No.", "none", "No, none.", "No none po", "None, none.", "wala", "Wala po.", "Wala, wala po.", "nothing",
    ])
    def test_income_none_answers(self, heard):
        from app.services.answer_service import understand_answer
        result = understand_answer("gross_annual_income", heard, spoken=True)
        assert result.ok, result.message
        assert result.value == "0" and result.display == "no income"

    def test_income_amount_still_read_after_a_no(self):
        from app.services.answer_service import understand_answer
        assert understand_answer("gross_annual_income", "No, 20 thousand a month").value == "240000"

    @pytest.mark.parametrize("heard,value", [
        ("Maria and Clara.", "Maria Clara"),
        ("My first name is Gerlie and Mae.", "Gerlie Mae"),
        ("Jose at Rizal", "Jose Rizal"),
        ("Andrea", "Andrea"),                       # "and" inside a name stays
        ("Atanacio", "Atanacio"),
        ("Andy Mae", "Andy Mae"),
    ])
    def test_given_name_joining_word_is_dropped(self, heard, value):
        from app.services.answer_service import understand_answer
        result = understand_answer("given_name", heard, spoken=True)
        assert result.ok, result.message
        assert result.value == value

    @pytest.mark.parametrize("heard,cleaned,value", [
        ("Jose at Rizal", "Jose Rizal", "Jose Rizal"),
        ("Jose Rizal", "Jose Rizal", "Jose Rizal"),
        ("My name is Maria", "Maria", "Maria"),
        ("Called Jose Rizal", "Called Jose Rizal", "Rizal"),
    ])
    def test_given_name_keeps_known_words_before_ner_span(self, heard, cleaned, value):
        from app.services import answer_service, name_lexicon
        from app.services.answer_service import Entity, understand_answer
        last = cleaned.split()[-1]
        start = cleaned.rfind(last)
        with patch.object(answer_service, "_ner_entities", return_value=[Entity("PER", last, 0.9, start, len(cleaned))]), \
                patch.object(name_lexicon, "_record_names", return_value={"given_name": {"Jose", "Maria"}}):
            result = understand_answer("given_name", heard, spoken=True)
        assert result.ok and result.value == value

    def test_yes_no_hint_has_no_navigation_words(self):
        from app.services.speech_service import FIELD_PROMPTS
        assert "back" not in FIELD_PROMPTS["confirm"].lower()

    # Fake addresses (no real residents): PSGC place names only.
    def test_region10_barangay_csv_counts(self):
        import csv
        from collections import Counter
        from app.services.barangay_gazetteer import REGION_CSV_PATH
        with open(REGION_CSV_PATH, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        per_town = Counter(r["town_psgc"] for r in rows)
        assert len(rows) == 2022 and per_town["1030500000"] == 80 and per_town["1001301000"] == 16
        imbatugs = {(r["town"], r["psgc_code"]) for r in rows if r["barangay"] == "Imbatug"}
        assert imbatugs == {("Baungon", "1001301012"), ("Cabanglasan", "1001322004")}
        assert {r["province"] for r in rows if r["town_psgc"] == "1030500000"} == {"Misamis Oriental"}

    @pytest.mark.parametrize("heard,town_psgc", [
        ("Cagayan de Oro", "1030500000"), ("Kagayan de oro city", "1030500000"), ("CDO", "1030500000"),
        ("Kagayan the Oro", "1030500000"),
        ("I live in Baungon", "1001301000"), ("Cabanglasan", "1001322000"), ("Valencia", "1001321000"),
    ])
    def test_address_town(self, heard, town_psgc):
        from app.services.answer_service import understand_answer
        result = understand_answer("address_town", heard, spoken=True)
        assert result.ok and not result.needs_confirmation, result
        assert result.normalized == town_psgc

    def test_unknown_town_offers_the_office_town(self):
        from app.services.answer_service import understand_answer
        result = understand_answer("address_town", "nails", spoken=True)
        assert result.ok and result.needs_confirmation
        assert result.value == "Baungon, Bukidnon" and result.normalized == "1001301000"
        assert "Baungon" in result.message

    def test_cdo_town_is_written_with_misamis_oriental(self):
        from app.services.answer_service import understand_answer
        assert understand_answer("address_town", "Cagayan de Oro").value == "Cagayan de Oro City, Misamis Oriental"

    @pytest.mark.parametrize("heard,town,value", [
        ("Imbatog", "1001301000", "Imbatug"),            # Baungon's Imbatug
        ("in Baedig", "1001301000", "Imbatug"),
        ("Barangay in Baedig", "1001301000", "Imbatug"),
        ("sa Lingating", "1001301000", "Lingating"),
        ("in Carmen", "1030500000", "Carmen"),
        ("in ten", "1030500000", "Barangay 10"),
        ("sa two", "1030500000", "Barangay 2"),
        ("Poblacion", "1001301000", "Imbatug"),          # its old name
        ("Imbatug", "1001322000", "Imbatug"),            # Cabanglasan's Imbatug: matched inside Cabanglasan
        ("Barangay Lapasan", "1030500000", "Lapasan"),   # a Cagayan de Oro barangay
        ("Bulua po", "1030500000", "Bulua"),
        ("Barangay ten", "1030500000", "Barangay 10"),   # numbered barangays of Cagayan de Oro
        ("Salim Balan", None, "Salimbalan"),             # no town: the office town's barangays
    ])
    def test_barangay_matched_inside_the_town(self, heard, town, value):
        from app.services.answer_service import understand_answer
        result = understand_answer("address_barangay", heard, spoken=True, town=town)
        assert result.ok and not result.needs_confirmation, result
        assert result.value == value

    def test_barangay_of_another_town_is_not_matched(self):
        from app.services.answer_service import understand_answer
        result = understand_answer("address_barangay", "Imbatug", spoken=True, town="1030500000")
        assert result.ok and result.needs_confirmation    # Imbatug is not in Cagayan de Oro: confirm as said
        assert result.display == "Barangay Imbatug"

    def test_barangay_outside_region_x_is_taken_as_said(self):
        from app.services.answer_service import understand_answer
        result = understand_answer("address_barangay", "Poblacion", spoken=True, town="0304926000")
        assert result.ok and result.needs_confirmation and result.value == "Poblacion"

    def test_barangay_with_unknown_town_is_taken_as_said(self):
        from app.services.answer_service import understand_answer
        result = understand_answer("address_barangay", "Poblacion", spoken=True, town="0")
        assert result.ok and result.needs_confirmation and result.value == "Poblacion"

    def test_unknown_town_barangay_hint_has_no_office_barangays(self):
        from app.services.speech_service import SpeechService
        assert SpeechService._build_prompt("address_barangay", "0") == "The name of the barangay."

    @pytest.mark.parametrize("heard,value", [
        ("Purok too", "Purok 2"), ("two", "Purok 2"), ("2", "Purok 2"), ("My purok is 5", "Purok 5"),
        ("Zone 1", "Zone 1"), ("123 Rizal Street", "123 Rizal Street"), ("none", ""), ("wala po", ""),
    ])
    def test_address_purok(self, heard, value):
        from app.services.answer_service import understand_answer
        result = understand_answer("address_purok", heard, spoken=True)
        assert result.ok and result.value == value

    @pytest.mark.parametrize("spoken,value", [
        ("Purok 4, Barangay Bulua, Cagayan de Oro City", "Purok 4, Barangay Bulua, Cagayan de Oro City, Misamis Oriental"),
        ("Zone 1 Carmen, Kagayan the Oro", "Zone 1, Barangay Carmen, Cagayan de Oro City, Misamis Oriental"),
        ("Purok 2 Lingating Baungon", "Purok 2, Barangay Lingating, Baungon, Bukidnon"),
        ("Purok 2 Imbatug", "Purok 2 Imbatug, Baungon, Bukidnon"),
        ("I live in Zone 3, Carmen, CDO", "Zone 3, Barangay Carmen, Cagayan de Oro City, Misamis Oriental"),
        ("Purok 1, Imbatug, Cabanglasan, Bukidnon", "Purok 1, Barangay Imbatug, Cabanglasan, Bukidnon"),
        ("Barangay seventeen, Cagayan de Oro", "Barangay 17, Cagayan de Oro City, Misamis Oriental"),
        # A joined 3-question answer is read back unchanged when checked again before issuing.
        ("Purok 2, Barangay Carmen, Cagayan de Oro City, Misamis Oriental",
         "Purok 2, Barangay Carmen, Cagayan de Oro City, Misamis Oriental"),
        ("Purok 2, Barangay Imbatug, Baungon, Bukidnon", "Purok 2, Barangay Imbatug, Baungon, Bukidnon"),
    ])
    def test_one_sentence_address_finds_the_town_first(self, spoken, value):
        from app.services.answer_service import understand_answer
        result = understand_answer("address", spoken)
        assert result.ok and not result.needs_confirmation, result
        assert result.value == value

    def test_office_province_never_added_to_another_town(self):
        from app.services.answer_service import understand_answer
        value = understand_answer("address", "Purok 5, Lapasan, Cagayan de Oro City").value
        assert "Baungon" not in value and "Bukidnon" not in value
        result = understand_answer("address", "Purok 1, Barangay Poblacion, San Jose, Batangas")
        assert result.needs_confirmation and "Baungon" not in result.value     # outside Region X: confirm as said

    @pytest.mark.parametrize("spoken", ["Purok 3, San Vicente", "Purok 2, Rizal Street, Imbatug", "Kalilangan"])
    def test_office_barangay_named_like_a_town_stays_in_the_office_town(self, spoken):
        from app.services.answer_service import understand_answer
        assert understand_answer("address", spoken).value.endswith("Baungon, Bukidnon")

    @pytest.mark.parametrize("heard,display", [
        ("Purok 3 Lapasan, Cagayan del Oro", "Cagayan de Oro City"),   # "Cagayan" (a province) inside a misheard name
        ("Cagayan the Oro City", "Cagayan de Oro City"),
        ("Purok 2, Carmen, Bohol", "Carmen"),                          # a town after a purok number
        ("Tuguegarao, Cagayan", "Tuguegarao City"),
    ])
    def test_find_town_misheard_and_province_names(self, heard, display):
        from app.services.place_gazetteer import find_town
        assert find_town(heard, home_province="Bukidnon")["place"].display == display

    def test_find_town_skips_a_barangay_name(self):
        from app.services.place_gazetteer import find_town
        assert find_town("Barangay Carmen", home_province="Bukidnon") is None   # Carmen is also a town elsewhere

    def test_split_address(self):
        from app.services.answer_service import split_address
        assert split_address("Purok 4, Bulua, Cagayan de Oro City") == {
            "town_psgc": "1030500000", "town": "Cagayan de Oro City", "town_said": True, "barangay": "Bulua",
            "rest": "Purok 4"}
        no_town = split_address("Purok 2 Imbatug")
        assert no_town["town_psgc"] == "1001301000" and not no_town["town_said"]   # the office town, assumed

    def test_barangay_hint_lists_the_town_said(self):
        from app.services.speech_service import SpeechService
        cdo = SpeechService._build_prompt("address_barangay", "1030500000")
        assert "Lapasan" in cdo and "Salimbalan" not in cdo
        assert "Salimbalan" in SpeechService._build_prompt("address_barangay", None)
        outside = SpeechService._build_prompt("address_barangay", "0304926000")
        assert "Salimbalan" not in outside

    def test_transcribe_sends_the_town_to_the_hint(self, client, staff_token):
        from app.services.speech_service import SpeechService
        with patch.object(SpeechService, "transcribe_bytes", return_value={
                "transcript": "Lapasan", "language": "en", "confidence": 0.9, "segments": [],
                "processing_time_seconds": 0.1}) as fake:
            r = client.post("/api/speech/transcribe", headers={"Authorization": f"Bearer {staff_token}"},
                            files={"audio_file": ("a.wav", b"RIFF0000", "audio/wav")},
                            data={"field_key": "address_barangay", "town": "1030500000"})
        assert r.status_code == 200
        assert fake.call_args.kwargs["town"] == "1030500000"

    def test_answer_endpoint_address_parts(self, client, staff_token):
        headers = {"Authorization": f"Bearer {staff_token}"}
        r = client.post("/api/extraction/answer", headers=headers, json={
            "field_key": "address_barangay", "transcript": "Lapasan", "spoken": True, "town": "1030500000"})
        assert r.status_code == 200 and r.json()["value"] == "Lapasan" and r.json()["display"] == "Barangay Lapasan"
        r = client.post("/api/extraction/answer", headers=headers, json={
            "field_key": "address_barangay", "transcript": "x", "town": "not-a-code"})
        assert r.status_code == 422


BUKIDNON = "1001300000"


class TestPlaceOfBirth:
    """Plan 001.8: province first, then a town of it; talk that is not a place is never taken as one."""

    @pytest.mark.parametrize("heard,value", [
        ("Bokidnon", "Bukidnon"), ("Misamis Orienthal", "Misamis Oriental"), ("Lanao del Norteh", "Lanao del Norte"),
        ("I was born in the province of Bukidnon", "Bukidnon"), ("Cebu", "Cebu"), ("Metro Manila", "Metro Manila"),
    ])
    def test_province_also_misheard(self, heard, value):
        from app.services.answer_service import understand_answer
        result = understand_answer("place_of_birth_province", heard, spoken=True)
        assert result.ok and result.value == value
        assert result.normalized.isdigit()           # the province's PSGC code: its town is asked next

    @pytest.mark.parametrize("heard,value", [
        ("Iligan", "Iligan City"), ("Kagayan de Oro", "Cagayan de Oro City"), ("Cagayan", "Cagayan de Oro City"),
        ("Pateros", "Pateros"), ("Cebu City", "Cebu City"), ("Malaybalay, Bukidnon", "Malaybalay City, Bukidnon"),
    ])
    def test_province_step_takes_a_whole_place(self, heard, value):
        from app.services.answer_service import understand_answer
        result = understand_answer("place_of_birth_province", heard, spoken=True)
        assert result.ok and result.value == value and result.normalized == ""   # done: no town question

    @pytest.mark.parametrize("heard", ["Cagayan province", "I was born in the province of Cagayan", "Probinsya sa Cagayan"])
    def test_province_word_makes_cagayan_the_province(self, heard):
        from app.services.answer_service import understand_answer
        result = understand_answer("place_of_birth_province", heard, spoken=True)
        assert result.ok and result.value == "Cagayan" and result.normalized.isdigit()

    def test_city_with_no_province_still_taken_at_the_town_step(self):
        # Cagayan de Oro has no province in the PSGC list, so it is not one of Misamis Oriental's towns.
        from app.services.answer_service import understand_answer
        result = understand_answer("place_of_birth_town", "Cagayan de Oro", spoken=True, province="1004300000")
        assert result.ok and result.value == "Cagayan de Oro City"

    @pytest.mark.parametrize("value", [
        "Quezon, Bukidnon", "Lantapan, Bukidnon", "Malaybalay City, Bukidnon", "Iligan City", "Cagayan de Oro City",
    ])
    def test_confirmed_place_survives_the_recheck_before_issuing(self, value):
        from app.services.answer_service import understand_answers
        assert understand_answers({"place_of_birth": value})["place_of_birth"].value == value

    def test_town_step_quezon_is_the_town_of_the_province(self):
        from app.services.answer_service import understand_answer
        assert understand_answer("place_of_birth_town", "Quezon", spoken=True, province=BUKIDNON).value == "Quezon, Bukidnon"

    @pytest.mark.parametrize("heard,province,value", [
        ("Valencia, Negros Oriental", BUKIDNON, "Valencia, Negros Oriental"),
        ("Quezon City", BUKIDNON, "Quezon City"),
        ("San Jose, Batangas", "0306900000", "San Jose, Batangas"),      # chosen: Tarlac
        ("Valencia", BUKIDNON, "Valencia City, Bukidnon"),                 # not named outright: the province's town
    ])
    def test_town_step_keeps_a_place_named_outright(self, heard, province, value):
        from app.services.answer_service import understand_answer
        assert understand_answer("place_of_birth_town", heard, spoken=True, province=province).value == value

    @pytest.mark.parametrize("field", ["place_of_birth_province", "place_of_birth"])
    def test_misheard_cagayan_province_is_the_province(self, field):
        from app.services.answer_service import understand_answer
        assert understand_answer(field, "Kagayan province", spoken=True).value == "Cagayan"

    def test_places_without_province_come_from_the_csv(self):
        import csv
        from app.services.place_gazetteer import DATA_FILE, places_without_province
        with open(DATA_FILE, newline="", encoding="utf-8") as f:
            blank = [r for r in csv.DictReader(f) if r["type"] != "province" and not r["province"]]
        assert len(places_without_province()) == len(blank) == 44
        assert sum(p.type == "city" for p in places_without_province()) == 35

    @pytest.mark.parametrize("heard,value", [
        ("Malaybalai", "Malaybalay City, Bukidnon"), ("Lantap", "Lantapan, Bukidnon"),
        ("Manulo", "Manolo Fortich, Bukidnon"), ("Valensya", "Valencia City, Bukidnon"),
        ("Impasugong", "Impasug-ong, Bukidnon"), ("Talakag", "Talakag, Bukidnon"),
    ])
    def test_town_near_miss_inside_the_province(self, heard, value):
        from app.services.answer_service import understand_answer
        result = understand_answer("place_of_birth_town", heard, spoken=True, province=BUKIDNON)
        assert result.ok and result.value == value

    def test_town_step_refuses_the_province_again(self):
        from app.services.answer_service import understand_answer
        result = understand_answer("place_of_birth_town", "Bukidnon", spoken=True, province=BUKIDNON)
        assert not result.ok and "Bukidnon" in result.message and "For example" in result.message
        assert len(result.suggestions) == 3
        assert all(s["value"].endswith(", Bukidnon") and s["province"] == "" for s in result.suggestions)

    @pytest.mark.parametrize("field,kwargs", [
        ("place_of_birth_province", {}), ("place_of_birth_town", {"province": BUKIDNON}),
        ("place_of_birth", {}),
    ])
    @pytest.mark.parametrize("junk", ["Sounds good", "Baby you are my sunshine tonight", "Okay okay", "I have a plan"])
    def test_talk_is_never_a_spoken_place(self, field, kwargs, junk):
        from app.services.answer_service import understand_answer
        from app.services.place_gazetteer import place_value, provinces, towns_in_province, places_without_province
        result = understand_answer(field, junk, spoken=True, **kwargs)
        assert not result.ok and not result.value
        real = ({p.display for p in provinces()} | {place_value(p) for p in places_without_province()}
                | {place_value(p) for p in towns_in_province(BUKIDNON)})
        assert result.suggestions and all(s["value"] in real for s in result.suggestions)

    def test_typed_place_not_in_the_list_is_kept_and_marked(self):
        from app.services.answer_service import understand_answer
        result = understand_answer("place_of_birth", "Sitio Somewhere Far")
        assert result.ok and result.value == "Sitio Somewhere Far" and result.needs_confirmation
        assert "not in the list" in result.warnings[0]

    def test_same_town_name_in_many_provinces_asks_the_province(self):
        from app.services.answer_service import understand_answer
        result = understand_answer("place_of_birth_province", "San Jose", spoken=True)
        assert not result.ok and "province" in result.message
        assert len(result.suggestions) == 3 and all(s["value"].startswith("San Jose, ") for s in result.suggestions)

    def test_nothing_heard_still_gives_places_to_tap(self):
        from app.services.answer_service import understand_answer
        province = understand_answer("place_of_birth_province", "", spoken=True)
        assert not province.ok and [s["value"] for s in province.suggestions][0] == "Bukidnon"
        town = understand_answer("place_of_birth_town", "", spoken=True, province=BUKIDNON)
        assert len(town.suggestions) == 3 and all(s["value"].endswith(", Bukidnon") for s in town.suggestions)

    def test_typed_place_not_in_the_list_offers_list_entries(self):
        from app.services.answer_service import understand_answer
        result = understand_answer("place_of_birth", "Okay okay")
        assert result.ok and result.needs_confirmation and result.suggestions
        assert all(", " in s["value"] or s["province"] for s in result.suggestions)

    def test_town_step_without_a_province_works_like_the_province_step(self):
        from app.services.answer_service import understand_answer
        assert understand_answer("place_of_birth_town", "Bokidnon", spoken=True).value == "Bukidnon"

    @pytest.mark.parametrize("heard,value", [("Sitio Mahogany", "Sitio Mahogany"), ("Prk 3", "Prk 3"),
                                             ("123 Mahogany", "123 Mahogany"), ("Mahogany Drive", "Mahogany Drive")])
    def test_purok_parsed(self, heard, value):
        from app.services.answer_service import understand_answer
        result = understand_answer("address_purok", heard, spoken=True)
        assert result.ok and result.value == value

    @pytest.mark.parametrize("junk", ["Sounds good", "Baby you are my sunshine", "I have 2 children", "I am 20 years old",
                                      "I drive a car", "20 years old"])
    def test_purok_talk_refused_with_an_example(self, junk):
        from app.services.answer_service import understand_answer
        result = understand_answer("address_purok", junk, spoken=True)
        assert not result.ok and "Purok 2" in result.message

    def test_purok_from_the_office_list(self, tmp_path):
        from app.services import answer_service
        purok_file = tmp_path / "office_puroks.csv"
        purok_file.write_text("name\nMahogany\nSampaguita\n", encoding="utf-8")
        with patch.object(answer_service, "PUROK_LIST_FILE", str(purok_file)):
            result = answer_service.understand_answer("address_purok", "Sampagita", spoken=True)
            assert result.ok and result.value == "Purok Sampaguita"
            assert not answer_service.understand_answer("address_purok", "Sounds good", spoken=True).ok

    def test_town_hint_lists_the_province_towns(self):
        from app.services.speech_service import SpeechService
        hint = SpeechService._build_prompt("place_of_birth_town", province=BUKIDNON)
        assert "Malaybalay" in hint and "Lantapan" in hint and "Iligan" not in hint
        assert SpeechService._build_prompt("place_of_birth_town") == "The town or city where you were born."
        assert "Bukidnon" in SpeechService._build_prompt("place_of_birth_province")

    def test_transcribe_sends_the_province_to_the_hint(self, client, staff_token):
        from app.services.speech_service import SpeechService
        with patch.object(SpeechService, "transcribe_bytes", return_value={
                "transcript": "Lantapan", "language": "en", "confidence": 0.9, "segments": [],
                "processing_time_seconds": 0.1}) as fake:
            r = client.post("/api/speech/transcribe", headers={"Authorization": f"Bearer {staff_token}"},
                            files={"audio_file": ("a.wav", b"RIFF0000", "audio/wav")},
                            data={"field_key": "place_of_birth_town", "province": BUKIDNON})
        assert r.status_code == 200
        assert fake.call_args.kwargs["province"] == BUKIDNON

    def test_answer_endpoint_birth_steps(self, client, staff_token):
        headers = {"Authorization": f"Bearer {staff_token}"}
        r = client.post("/api/extraction/answer", headers=headers, json={
            "field_key": "place_of_birth_province", "transcript": "Bokidnon", "spoken": True})
        assert r.status_code == 200 and r.json()["normalized"] == BUKIDNON
        r = client.post("/api/extraction/answer", headers=headers, json={
            "field_key": "place_of_birth_town", "transcript": "Sounds good", "spoken": True, "province": BUKIDNON})
        assert r.json()["ok"] is False and len(r.json()["suggestions"]) == 3
        r = client.post("/api/extraction/answer", headers=headers, json={
            "field_key": "place_of_birth_town", "transcript": "Lantap", "spoken": True, "province": BUKIDNON})
        assert r.json()["value"] == "Lantapan, Bukidnon"
        r = client.post("/api/extraction/answer", headers=headers, json={
            "field_key": "place_of_birth_town", "transcript": "x", "province": "not-a-code"})
        assert r.status_code == 422


# ─── SUS Survey ────────────────────────────────────────────────────────────────

class TestSusSurvey:
    ANSWERS = dict(zip([f"q{i}" for i in range(1, 11)], [4, 2, 5, 1, 4, 2, 5, 2, 4, 1]))

    def survey(self, **overrides):
        return {
            "respondent_name": "  ", "role": "citizen_applicant", "computer_experience": "basic",
            "features_used": ["voice_input", "qr_prefill", "voice_input"], **self.ANSWERS,
            "comments": "Easy to follow", "consent_given": True, "platform": "web", **overrides,
        }

    def test_submit_without_login_and_score_hidden(self, client, staff_token):
        resp = client.post("/api/sus/responses", json=self.survey())
        assert resp.status_code == 201
        assert "sus_score" not in resp.json()

        rows = client.get("/api/sus/responses", headers={"Authorization": f"Bearer {staff_token}"}).json()
        row = next(r for r in rows if r["id"] == resp.json()["id"])
        assert row["sus_score"] == 85.0                      # the reference test case
        assert row["respondent_name"] is None                # blank name stored as empty
        assert row["features_used"] == ["voice_input", "qr_prefill"]
        assert row["synced_to_sheets"] is False

    def test_rejects_invalid_answers(self, client):
        for bad in ({"consent_given": False}, {"q3": 6}, {"q10": 0}, {"role": "mayor"},
                    {"features_used": ["hacking"]}, {"platform": "desktop"}, {"transaction_id": 999999}):
            assert client.post("/api/sus/responses", json=self.survey(**bad)).status_code == 422, bad
        missing = self.survey()
        del missing["q7"]
        assert client.post("/api/sus/responses", json=missing).status_code == 422

    def test_links_to_transaction(self, client, staff_token, db):
        from app.models.issuance import CTCIssuance
        issuance_id = db.query(CTCIssuance.id).first()[0]
        resp = client.post("/api/sus/responses", json=self.survey(transaction_id=issuance_id, platform="mobile"))
        assert resp.status_code == 201

    def test_reading_needs_staff(self, client, db):
        from app.models.staff_user import StaffUser, RoleEnum
        assert client.get("/api/sus/responses").status_code == 401
        db.add(StaffUser(username="sus_viewer", hashed_password=hash_password("testpass123"),
                         full_name="Survey Viewer", role=RoleEnum.VIEWER))
        db.commit()
        viewer = {"Authorization": f"Bearer {create_access_token(data={'sub': 'sus_viewer', 'type': 'staff'})}"}
        assert client.get("/api/sus/responses", headers=viewer).status_code == 403
        citizen = {"Authorization": f"Bearer {create_access_token(data={'sub': 'admin_test', 'type': 'citizen'})}"}
        assert client.get("/api/sus/responses", headers=citizen).status_code == 401

    def test_summary_and_resync_permissions(self, client, staff_token, admin_token):
        summary = client.get("/api/sus/summary", headers={"Authorization": f"Bearer {staff_token}"}).json()
        assert summary["total"] >= 2 and summary["pending_sync"] == summary["total"]
        assert summary["mean_sus_score"] == 85.0
        assert client.post("/api/sus/resync", headers={"Authorization": f"Bearer {staff_token}"}).status_code == 403
        # The test database is SQLite, which has no Edge Function to call.
        assert client.post("/api/sus/resync", headers={"Authorization": f"Bearer {admin_token}"}).status_code == 503


# ─── Citizen sign-up and email verification switch ────────────────────────────

def _citizen_payload(username: str) -> dict:
    return {
        "first_name": "Juana",
        "last_name": "Testcase",
        "username": username,
        "password": "Testpass1!",
        "confirm_password": "Testpass1!",
        "profile_picture": "data:image/png;base64,iVBORw0KGgoAAAANSUhEUg==",
    }


def _citizen_login(client, username: str):
    return client.post("/api/citizen/login", json={"username": username, "password": "Testpass1!"})


class TestCitizenSignup:
    def test_verification_off_register_then_login_succeeds(self, client):
        from app.core.config import settings
        with patch.object(settings, "CITIZEN_EMAIL_VERIFICATION", False):
            resp = client.post("/api/citizen/register", json=_citizen_payload("citizen_noverify"))
            assert resp.status_code == 201
            assert resp.json()["requires_verification"] is False
            login = _citizen_login(client, "citizen_noverify")
        assert login.status_code == 200
        assert login.json()["access_token"]

    def test_verification_on_register_then_login_refused(self, client):
        from app.core.config import settings
        with patch.object(settings, "CITIZEN_EMAIL_VERIFICATION", True):
            resp = client.post("/api/citizen/register", json=_citizen_payload("citizen_mustverify"))
            assert resp.status_code == 201
            assert resp.json()["requires_verification"] is True
            login = _citizen_login(client, "citizen_mustverify")
        assert login.status_code == 403
        assert "verify your email" in login.json()["detail"]

    def test_safe_defaults(self):
        from app.core.config import Settings
        assert Settings.model_fields["CITIZEN_EMAIL_VERIFICATION"].default is True
        assert Settings.model_fields["DEBUG"].default is False


# ─── Health Check ──────────────────────────────────────────────────────────────

class TestHealth:
    def test_root_health(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "running"

    def test_api_health(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200

    def test_protected_route_without_token(self, client):
        resp = client.get("/api/auth/me")
        assert resp.status_code == 401


class TestWhisperEngine:
    """Plan 002: WHISPER_ENGINE switches openai-whisper / faster-whisper behind the same interface."""

    @staticmethod
    def _fake_faster(calls):
        """A stand-in faster_whisper module that records how it was called (no model is loaded)."""
        import types

        class WhisperModel:
            def __init__(self, name, **kwargs):
                calls.append(("load", name, kwargs))

            def transcribe(self, audio, **kwargs):
                calls.append(("transcribe", audio, kwargs))
                segment = types.SimpleNamespace(id=0, start=0.0, end=1.2, text=" Dela Cruz po.", avg_logprob=-0.2,
                                                compression_ratio=1.1)
                return iter([segment]), types.SimpleNamespace(language="en")

        return types.SimpleNamespace(WhisperModel=WhisperModel)

    def test_openai_is_the_default(self):
        from app.services import speech_service
        assert speech_service.settings.WHISPER_ENGINE == "openai"
        assert speech_service._engine() == "openai"

    def test_faster_engine_same_keys_and_explicit_options(self):
        import sys
        import numpy as np
        from app.services import speech_service
        from app.services.speech_service import SpeechService
        calls = []
        with patch.dict(sys.modules, {"faster_whisper": self._fake_faster(calls)}), \
                patch.object(speech_service.settings, "WHISPER_ENGINE", "faster"), \
                patch.object(speech_service.settings, "WHISPER_FIELD_MODELS", False), \
                patch.dict(SpeechService._models, clear=True):
            out = SpeechService.transcribe_file(np.zeros(16000, dtype=np.float32), "surname")
        assert set(out) == {"transcript", "dropped", "language", "confidence", "segments", "processing_time_seconds",
                            "passes"}
        assert out["transcript"] == "Dela Cruz po." and out["language"] == "en" and out["dropped"] is None
        assert out["segments"][0]["text"] == "Dela Cruz po." and 0 < out["confidence"] < 1
        (_, name, load_kwargs), (_, _, options) = calls
        assert name == "base" and load_kwargs == {"device": "cpu", "compute_type": "int8"}
        # faster-whisper's own defaults differ (beam 5, previous text on, no VAD): every one is set here.
        assert options["beam_size"] == 1 and options["condition_on_previous_text"] is False
        assert options["language"] == "en" and options["vad_filter"] is True
        assert options["initial_prompt"] == speech_service.FIELD_PROMPTS["surname"]
        assert options["temperature"] == speech_service.FALLBACK_TEMPERATURES

    def test_confirm_model_rule_is_the_same_on_faster(self):
        import sys
        import numpy as np
        from app.services import speech_service
        from app.services.speech_service import SpeechService
        calls = []
        with patch.dict(sys.modules, {"faster_whisper": self._fake_faster(calls)}), \
                patch.object(speech_service.settings, "WHISPER_ENGINE", "faster"), \
                patch.dict(SpeechService._models, clear=True):
            SpeechService.transcribe_file(np.zeros(16000, dtype=np.float32), "confirm")
        assert calls[0][1] == "tiny"

    def test_faster_warm_up_runs_the_model(self):
        import sys
        from app.services import speech_service
        from app.services.speech_service import SpeechService
        calls = []
        with patch.dict(sys.modules, {"faster_whisper": self._fake_faster(calls)}), \
                patch.object(speech_service.settings, "WHISPER_ENGINE", "faster"), \
                patch.dict(SpeechService._models, clear=True):
            SpeechService.warm_up()
        decodes = [kwargs for kind, _, kwargs in calls if kind == "transcribe"]
        # tiny, base and medium each load and decode once, with the VAD off: it would cut the silent clip away.
        assert sorted(name for kind, name, _ in calls if kind == "load") == ["base", "medium", "tiny"]
        assert len(decodes) == 3 and all(kwargs["vad_filter"] is False for kwargs in decodes)

    def test_warm_up_with_the_table_off_loads_only_the_main_model(self):
        import sys
        from app.services import speech_service
        from app.services.speech_service import SpeechService
        calls = []
        with patch.dict(sys.modules, {"faster_whisper": self._fake_faster(calls)}), \
                patch.object(speech_service.settings, "WHISPER_ENGINE", "faster"), \
                patch.object(speech_service.settings, "WHISPER_FIELD_MODELS", False), \
                patch.dict(SpeechService._models, clear=True):
            SpeechService.warm_up()
        assert [name for kind, name, _ in calls if kind == "load"] == ["base"]

    def test_with_the_table_off_every_question_uses_the_global_model_and_beam(self):
        from app.services import speech_service
        from app.services.speech_service import SpeechService
        with patch.object(speech_service.settings, "WHISPER_FIELD_MODELS", False),                 patch.object(speech_service.settings, "WHISPER_MODEL", "base"),                 patch.object(speech_service.settings, "WHISPER_BEAM_SIZE", 1):
            for key in ("confirm", "surname", "review", "spelling", None):
                assert SpeechService.model_and_beam_for(key) == ("base", 1)

    # ── Plan 006: one model and beam per question (Ralph's PLAN 6 table) ──
    TABLE = [
        ("surname", "medium", 5), ("given_name", "medium", 5), ("middle_name", "medium", 5),
        ("address", "medium", 5), ("address_town", "medium", 5),
        ("address_barangay", "medium", 5), ("address_purok", "medium", 5),
        ("date", "medium", 5), ("sex", "medium", 5), ("civil_status", "medium", 5), ("citizenship", "medium", 5),
        ("place_of_birth", "medium", 5), ("place_of_birth_province", "medium", 5),
        ("place_of_birth_town", "medium", 5),
        ("height", "medium", 5), ("weight", "medium", 5), ("occupation", "medium", 5),
        ("confirm", "tiny", 5), ("gross_annual_income", "medium", 5), ("icr_no", "medium", 5),
        ("review", "tiny", 5), ("spelling", "medium", 5), ("missing_choice", "base", 1),
        (None, "base", 1), ("unknown_key", "base", 1),
    ]

    @staticmethod
    def _fake_openai(calls):
        def load_model(name):
            model = MagicMock()

            def transcribe(audio, **kwargs):
                calls.append(("transcribe", name, kwargs))
                return {"text": " Dela Cruz po.", "language": "en", "segments": []}
            model.transcribe.side_effect = transcribe
            calls.append(("load", name, {}))
            return model
        return load_model

    @pytest.mark.parametrize("engine", ["faster", "openai"])
    @pytest.mark.parametrize("capture", ["file", "bytes", "numpy"])
    @pytest.mark.parametrize("field_key,model,beam", TABLE)
    def test_each_question_uses_its_models_row(self, field_key, model, beam, capture, engine):
        import sys
        import numpy as np
        from app.services import speech_service
        from app.services.speech_service import SpeechService
        calls = []
        samples = np.zeros(16000, dtype=np.float32)
        with patch.dict(sys.modules, {"faster_whisper": self._fake_faster(calls)}), \
                patch.object(speech_service.settings, "WHISPER_ENGINE", engine), \
                patch.object(speech_service.settings, "WHISPER_MODEL", "base"), \
                patch.object(speech_service.settings, "WHISPER_BEAM_SIZE", 1), \
                patch.object(speech_service.whisper, "load_model", side_effect=self._fake_openai(calls)), \
                patch.object(speech_service, "_decode_wav", return_value=samples), \
                patch.dict(SpeechService._models, clear=True):
            if capture == "numpy":
                SpeechService.transcribe_numpy(samples, field_key=field_key)
            elif capture == "bytes":
                SpeechService.transcribe_bytes(b"test-audio", field_key=field_key)
            else:
                SpeechService.transcribe_file(samples, field_key=field_key)
        loads = [c for c in calls if c[0] == "load"]
        decodes = [c for c in calls if c[0] == "transcribe"]
        # "Dela Cruz po." is not a plain yes, so `review` is decoded a second time (Medium); the others once.
        assert loads[0][1] == model and len(decodes) == (2 if field_key == "review" else 1)
        options = decodes[0][2]   # the question's own row (first pass)
        if engine == "faster":
            assert options["beam_size"] == beam and options["max_new_tokens"] == speech_service.MAX_ANSWER_TOKENS
        else:
            assert options.get("beam_size", 1) == beam and options["sample_len"] == speech_service.MAX_ANSWER_TOKENS
        assert (options["temperature"] == 0.0) == (beam > 1)   # beam search decodes once, no ladder

    def _review(self, texts, field_enabled=True):
        """Run a `review` answer whose decodes return `texts` in turn; returns (result, models decoded)."""
        import sys
        import types
        import numpy as np
        from app.services import speech_service
        from app.services.speech_service import SpeechService
        decoded, queue = [], list(texts)

        class WhisperModel:
            def __init__(self, name, **kwargs):
                self.name = name

            def transcribe(self, audio, **kwargs):
                decoded.append((self.name, kwargs["beam_size"]))
                text = queue.pop(0)
                segments = [types.SimpleNamespace(id=0, start=0.0, end=1.0, text=text, avg_logprob=-0.2,
                                                  compression_ratio=1.1)] if text else []
                return iter(segments), types.SimpleNamespace(language="en")

        with patch.dict(sys.modules, {"faster_whisper": types.SimpleNamespace(WhisperModel=WhisperModel)}), \
                patch.object(speech_service.settings, "WHISPER_ENGINE", "faster"), \
                patch.object(speech_service.settings, "WHISPER_FIELD_MODELS", field_enabled), \
                patch.dict(SpeechService._models, clear=True):
            out = SpeechService.transcribe_file(np.zeros(16000, dtype=np.float32), "review")
        return out, decoded

    def test_review_plain_yes_is_one_tiny_decode(self):
        out, decoded = self._review([" Yes."])
        assert out["transcript"] == "Yes." and decoded == [("tiny", 5)]

    def test_review_correction_is_decoded_again_with_medium(self):
        out, decoded = self._review([" Yes my surname is Cruz.", " Yes, my surname is Cruz."])
        assert out["transcript"] == "Yes, my surname is Cruz."
        assert decoded == [("tiny", 5), ("medium", 5)]

    def test_review_silence_is_one_decode(self):
        out, decoded = self._review([""])
        assert out["transcript"] == "" and decoded == [("tiny", 5)]

    def test_review_with_the_table_off_is_one_decode_and_never_loads_medium(self):
        out, decoded = self._review([" Change surname to Cruz."], field_enabled=False)
        assert decoded == [("base", 1)]

    @pytest.mark.parametrize("text,expected", [
        ("Yes.", True), ("yes po", True), ("Opo", True), ("Okay", True), ("Correct.", True),
        ("Yes, my surname is Cruz.", False), ("Yes but change the address", False), ("No", False),
        ("Yes male", False), ("", False), ("Yes yes yes yes", False),
    ])
    def test_is_plain_yes_is_a_whole_reply(self, text, expected):
        from app.services.speech_service import is_plain_yes
        assert is_plain_yes(text) is expected

    def test_faster_missing_falls_back_to_openai(self, caplog):
        import sys
        import numpy as np
        from app.services import speech_service
        from app.services.speech_service import SpeechService
        openai_model = MagicMock()
        openai_model.transcribe.return_value = {"text": " Female.", "language": "en", "segments": []}
        caplog.set_level("ERROR")
        # A None entry in sys.modules makes "import faster_whisper" raise ImportError.
        with patch.dict(sys.modules, {"faster_whisper": None}), \
                patch.object(speech_service.settings, "WHISPER_ENGINE", "faster"), \
                patch.object(speech_service, "_engine_warned", False), \
                patch.object(speech_service.whisper, "load_model", return_value=openai_model) as load, \
                patch.dict(SpeechService._models, clear=True):
            out = SpeechService.transcribe_file(np.zeros(16000, dtype=np.float32), "sex")
            SpeechService.transcribe_file(np.zeros(16000, dtype=np.float32), "sex")
        assert out["transcript"] == "Female." and load.call_count == 1
        errors = [r for r in caplog.records if "Using openai-whisper instead" in r.getMessage()]
        assert len(errors) == 1   # said once, not on every answer

    def test_beam_search_has_no_temperature_ladder(self):
        from app.services import speech_service
        with patch.object(speech_service.settings, "WHISPER_BEAM_SIZE", 5):
            openai_options = speech_service._decode_options(None)
            faster_options = speech_service._faster_options(None)
        assert openai_options["beam_size"] == 5 and openai_options["temperature"] == 0.0
        assert faster_options["beam_size"] == 5 and faster_options["temperature"] == 0.0
        assert speech_service._decode_options(None)["temperature"] == speech_service.FALLBACK_TEMPERATURES

    def test_answer_length_capped_and_repetition_dropped(self):
        from app.services import speech_service
        from app.services.speech_service import MAX_ANSWER_TOKENS, _clean_transcript
        # The bake-off's small/beam-5 loop: one address became 157 words, 7 distinct.
        assert speech_service._decode_options(None)["sample_len"] == MAX_ANSWER_TOKENS
        assert speech_service._faster_options(None)["max_new_tokens"] == MAX_ANSWER_TOKENS
        loop = "Purok 3, Barangay Uno, " * 20
        assert _clean_transcript(loop, [{"compression_ratio": 9.5}]) == ""
        assert _clean_transcript("Purok 3, Barangay Uno.", [{"compression_ratio": 1.1}]) == "Purok 3, Barangay Uno."
        assert _clean_transcript("Female.") == "Female."   # no segments: only the silence phrases are dropped

    def test_models_are_cached_per_engine(self):
        import sys
        from app.services import speech_service
        from app.services.speech_service import SpeechService
        calls = []
        with patch.dict(sys.modules, {"faster_whisper": self._fake_faster(calls)}), \
                patch.object(speech_service.whisper, "load_model", return_value=MagicMock()) as load, \
                patch.dict(SpeechService._models, clear=True):
            with patch.object(speech_service.settings, "WHISPER_ENGINE", "faster"):
                faster_model = SpeechService.get_model("base")
                assert SpeechService.get_model("base") is faster_model
            openai_model = SpeechService.get_model("base")
        assert openai_model is not faster_model and load.call_count == 1 and len(calls) == 1


class TestHintEcho:
    """Plan 002.2: Whisper reading the question's hint back is never an answer (made-up text only)."""

    @pytest.mark.parametrize("key", ["sex", "civil_status", "confirm", "surname", "given_name", "middle_name",
                                     "address_purok", "missing_choice", "citizenship", "review"])
    def test_the_hint_read_back_is_dropped(self, key):
        from app.services.speech_service import FIELD_PROMPTS, _clean_answer
        prompt = FIELD_PROMPTS[key]
        assert _clean_answer(prompt, [], prompt, key) == ("", "echo")
        assert _clean_answer(prompt.rstrip(".").lower(), [], prompt, key) == ("", "echo")   # no punctuation

    @pytest.mark.parametrize("key,spoken", [
        ("sex", "Male or female."), ("sex", "Lalaki or babae."), ("confirm", "Yes or no."),
        ("confirm", "Opo, oo, hindi."), ("civil_status", "Single, married, widowed, or separated."),
        ("surname", "Lagbas po. Ybañez po."), ("address_purok", "Purok 2. Zone 1."),
    ])
    def test_a_run_of_hint_sentences_is_dropped(self, key, spoken):
        from app.services.speech_service import FIELD_PROMPTS, _clean_answer
        assert _clean_answer(spoken, [], FIELD_PROMPTS[key], key) == ("", "echo")

    @pytest.mark.parametrize("key,spoken", [
        ("surname", "Lagbas po."), ("given_name", "Jessa po."), ("middle_name", "Ybañez."),
        ("address_purok", "Purok 2."), ("address_purok", "Zone 1."),
        ("missing_choice", "Type."), ("missing_choice", "Say it again."),
        ("sex", "Female."), ("sex", "Lalaki po."), ("civil_status", "Married."), ("confirm", "Oo."),
        ("confirm", "Yes."), ("confirm", "Hindi."), ("citizenship", "Filipino."),
        ("date", "May 10, 1990."), ("height", "160 centimeters."),
    ])
    def test_a_real_answer_that_is_also_a_hint_example_is_kept(self, key, spoken):
        from app.services.speech_service import FIELD_PROMPTS, _clean_answer
        assert _clean_answer(spoken, [], FIELD_PROMPTS[key], key) == (spoken, None)

    def test_a_barangay_answer_is_not_the_barangay_hint(self):
        from app.services.speech_service import SpeechService, _clean_answer
        prompt = SpeechService._build_prompt("address_barangay", None)
        spoken = "Barangay Imbatug, Baungon, Bukidnon."
        assert _clean_answer(spoken, [], prompt, "address_barangay") == (spoken, None)

    @pytest.mark.parametrize("key,spoken,kept", [
        ("sex", "Male or female. Lalaki or babae. Female.", "Female."),
        ("confirm", "Yes or no. Yes po.", "Yes po."),
        ("surname", "Lagbas po. Ybañez po. Cruz po.", "Cruz po."),
    ])
    def test_hint_then_answer_keeps_the_answer(self, key, spoken, kept):
        from app.services.speech_service import FIELD_PROMPTS, _clean_answer
        assert _clean_answer(spoken, [], FIELD_PROMPTS[key], key) == (kept, None)

    def test_codex_review_echo_boundaries(self):
        """Codex verify round 1 (plan 002.2): three cases the first filter got wrong."""
        from app.services.speech_service import FIELD_PROMPTS as P, _clean_answer
        # The whole surname hint read back with a small change is still the hint, never an answer.
        assert _clean_answer(P["surname"].replace("Tagaylo.", "Tagaylo po."), [], P["surname"], "surname") == ("", "echo")
        # A real purok after the whole hint is kept, though it looks like a hint example (Purok 2).
        for purok in ("Purok 5.", "Purok 3.", "Zone 4."):
            assert _clean_answer(f"{P['address_purok']} {purok}", [], P["address_purok"], "address_purok") == (purok, None)
        # One answer-like hint sentence said alone is an answer, also the longest one.
        said = "My middle name is Ybañez."
        assert _clean_answer(said, [], P["middle_name"], "middle_name") == (said, None)
        # The hint read back with no sentence breaks is still the hint.
        flat = "male or female lalaki or babae"
        assert _clean_answer(flat, [], P["sex"], "sex") == ("", "echo")

    def test_codex_round_2_merged_hint_sentences(self):
        """Codex verify round 2 (plan 002.2): Whisper merging two hint sentences must not leak an example."""
        from app.services.speech_service import FIELD_PROMPTS as P, _clean_answer
        merged = "Jessa po. My first name is Rodel Mylene po. Gerlie Mae."
        assert _clean_answer(merged, [], P["given_name"], "given_name") == ("", "echo")
        assert _clean_answer(merged + " Ana po.", [], P["given_name"], "given_name") == ("Ana po.", None)
        flat = P["surname"].replace(".", "")                     # no sentence breaks at all
        assert _clean_answer(flat, [], P["surname"], "surname") == ("", "echo")
        assert _clean_answer(flat + " Cruz", [], P["surname"], "surname") == ("Cruz", None)

    def test_drop_reasons(self):
        from app.services.speech_service import _clean_answer
        assert _clean_answer("Thank you.", []) == ("", "silence-phrase")
        assert _clean_answer("Purok 3, " * 20, [{"compression_ratio": 9.5}]) == ("", "repetition")
        assert _clean_answer("Female.", []) == ("Female.", None)

    @pytest.mark.parametrize("dropped,reason", [
        ("echo", "dropped-echo"), ("repetition", "dropped-repetition"), ("silence-phrase", "whisper-empty"),
        (None, "whisper-empty"),
    ])
    def test_the_422_says_why_it_was_empty(self, client, staff_token, dropped, reason):
        from app.services.speech_service import SpeechService
        with patch.object(SpeechService, "transcribe_bytes", return_value={
                "transcript": "", "dropped": dropped, "language": "en", "confidence": 0.0, "segments": [],
                "processing_time_seconds": 0.1}):
            r = client.post("/api/speech/transcribe", headers={"Authorization": f"Bearer {staff_token}"},
                            files={"audio_file": ("a.wav", b"RIFF0000", "audio/wav")}, data={"field_key": "sex"})
        assert r.status_code == 422
        assert r.headers["X-Empty-Reason"] == reason
        assert "Could not detect any speech" in r.json()["detail"]

    def test_a_kept_transcript_has_no_drop_reason_in_the_response(self, client, staff_token):
        from app.services.speech_service import SpeechService
        with patch.object(SpeechService, "transcribe_bytes", return_value={
                "transcript": "Female.", "dropped": None, "language": "en", "confidence": 0.9, "segments": [],
                "processing_time_seconds": 0.1}):
            r = client.post("/api/speech/transcribe", headers={"Authorization": f"Bearer {staff_token}"},
                            files={"audio_file": ("a.wav", b"RIFF0000", "audio/wav")}, data={"field_key": "sex"})
        assert r.status_code == 200 and r.json()["transcript"] == "Female." and "dropped" not in r.json()

    @pytest.mark.parametrize("spoken", [
        "Single, married, widowed or separated.", "single or married", "married but separated",
        "widowed, married",
    ])
    def test_civil_status_refuses_two_statuses(self, spoken):
        from app.services.answer_service import understand_answer
        result = understand_answer("civil_status", spoken)
        assert not result.ok and "only one" in result.message

    @pytest.mark.parametrize("spoken,value", [
        ("not married", "Single"), ("never married po", "Single"), ("single", "Single"),
        ("married", "Married"), ("hiwalay na", "Separated"), ("biyuda", "Widowed"),
    ])
    def test_civil_status_one_status_still_works(self, spoken, value):
        from app.services.answer_service import understand_answer
        result = understand_answer("civil_status", spoken)
        assert result.ok and result.value == value


BAUNGON = "1001301000"
CDO = "1030500000"


class TestPickers:
    """Plan 002.1: the lists behind the typing pickers, and what /forms/issue checks (fake data only)."""

    def test_birth_places_search_the_voice_list(self, client, staff_token):
        r = client.get("/api/places/birth", params={"q": "malaybalay"}, headers=_auth(staff_token))
        assert r.status_code == 200
        assert r.json()[0]["value"] == "Malaybalay City, Bukidnon" and r.json()[0]["code"]
        # A city with no province is in the list too, written without one (as the voice path writes it).
        values = [o["value"] for o in client.get("/api/places/birth", params={"q": "iligan"},
                                                 headers=_auth(staff_token)).json()]
        assert "Iligan City" in values
        assert client.get("/api/places/birth", params={"q": ""}, headers=_auth(staff_token)).json() == []

    def test_every_province_less_place_can_be_found(self):
        from app.services import place_gazetteer as g
        loose = g.places_without_province()
        assert len(loose) == 44
        for place in loose:
            assert place in g.search_birth_places(place.name, limit=50), place.name

    def test_towns_and_barangays_are_written_like_the_voice_path(self, client, staff_token):
        from app.services.answer_service import understand_answer
        towns = {o["code"]: o["value"] for o in client.get("/api/places/towns", headers=_auth(staff_token)).json()}
        assert towns[BAUNGON] == "Baungon, Bukidnon"
        assert towns[CDO] == "Cagayan de Oro City, Misamis Oriental"
        assert understand_answer("address_town", "Cagayan de Oro").value == towns[CDO]   # the same text
        brgys = client.get(f"/api/places/towns/{BAUNGON}/barangays", headers=_auth(staff_token)).json()
        imbatug = next(b for b in brgys if b["name"] == "Imbatug")
        assert imbatug["label"] == "Barangay Imbatug"
        assert client.get("/api/places/towns/0304926000/barangays", headers=_auth(staff_token)).status_code == 404

    def test_lists_need_a_staff_login(self, client):
        assert client.get("/api/places/towns").status_code in (401, 403)
        assert client.get("/api/places/birth", params={"q": "malay"}).status_code in (401, 403)

    @pytest.mark.parametrize("changes,field", [
        ({"height_cm": 400}, "height_cm"), ({"height_cm": 49}, "height_cm"), ({"weight_kg": 301}, "weight_kg"),
        ({"weight_kg": "heavy"}, "weight_kg"),
    ])
    def test_height_and_weight_outside_the_voice_limits_are_refused(self, client, staff_token, changes, field):
        before = _row_counts()
        r = _issue(client, staff_token, **changes)
        assert r.status_code == 422
        assert [item["loc"][-1] for item in r.json()["detail"]] == [field]
        assert _row_counts() == before                                   # nothing written

    @pytest.mark.parametrize("changes", [
        {"height_cm": 0, "weight_kg": 0}, {"height_cm": None, "weight_kg": ""}, {"height_cm": 50, "weight_kg": 300},
        {"height_cm": 250, "weight_kg": 20},
    ])
    def test_blank_or_boundary_height_and_weight_are_issued(self, client, staff_token, changes):
        assert _issue(client, staff_token, **changes).status_code == 201

    def test_a_picked_place_of_birth_must_be_that_list_place(self, client, staff_token):
        from app.services import place_gazetteer as g
        code = g.search_birth_places("malaybalay")[0].psgc
        ok = _issue(client, staff_token, place_of_birth="Malaybalay City, Bukidnon", place_of_birth_code=code)
        assert ok.status_code == 201
        # Title-cased by the web ("De" for "de"): the same place.
        cdo = g.search_birth_places("cagayan de oro")[0].psgc
        assert _issue(client, staff_token, place_of_birth="Cagayan De Oro City", place_of_birth_code=cdo).status_code == 201
        bad = _issue(client, staff_token, place_of_birth="Somewhere Else", place_of_birth_code=code)
        assert bad.status_code == 422 and bad.json()["detail"][0]["loc"][-1] == "place_of_birth"
        assert _issue(client, staff_token, place_of_birth="Nowhere", place_of_birth_code="999").status_code == 422

    def test_typed_free_text_is_not_list_checked(self, client, staff_token):
        # "Not in the list": no code is sent, as from the voice path and older phones.
        assert _issue(client, staff_token, place_of_birth="Tokyo, Japan").status_code == 201
        assert _issue(client, staff_token, address="Purok 1, Some Town, Lanao del Sur").status_code == 201

    def test_a_picked_barangay_must_be_in_the_picked_town(self, client, staff_token):
        good = "Purok 7, Barangay Imbatug, Baungon, Bukidnon"
        assert _issue(client, staff_token, address=good, address_town_code=BAUNGON,
                      address_barangay="Imbatug").status_code == 201
        wrong_town = _issue(client, staff_token, address="Barangay Lapasan, Baungon, Bukidnon",
                            address_town_code=BAUNGON, address_barangay="Lapasan")
        assert wrong_town.status_code == 422 and wrong_town.json()["detail"][0]["loc"][-1] == "address"
        other_text = _issue(client, staff_token, address="Purok 7, Barangay Imbatug, Talakag, Bukidnon",
                            address_town_code=BAUNGON, address_barangay="Imbatug")
        assert other_text.status_code == 422
        assert _issue(client, staff_token, address=good, address_town_code="0304926000").status_code == 422

    def test_the_picked_barangay_is_a_whole_part_of_the_address(self, client, staff_token):
        """Codex verify (plan 002.1): never a substring match."""
        r = _issue(client, staff_token, address="Barangay 10, Cagayan de Oro City, Misamis Oriental",
                   address_town_code=CDO, address_barangay="Barangay 1")
        assert r.status_code == 422 and r.json()["detail"][0]["loc"][-1] == "address"
        r = _issue(client, staff_token, address="Barangay Imbatugon, Baungon, Bukidnon",
                   address_town_code=BAUNGON, address_barangay="Imbatug")
        assert r.status_code == 422
        # The town must be the end of the address, not just somewhere in it.
        r = _issue(client, staff_token, address="Barangay Imbatug, Baungon, Bukidnon, Some Other Place",
                   address_town_code=BAUNGON, address_barangay="Imbatug")
        assert r.status_code == 422
        # Title-cased by the web, with or without a purok: still the pick.
        for address in ("Barangay Imbatug, Baungon, Bukidnon", "Purok 7, Barangay Imbatug, Baungon, Bukidnon"):
            assert _issue(client, staff_token, address=address, address_town_code=BAUNGON,
                          address_barangay="Imbatug").status_code == 201
        cdo = _issue(client, staff_token, address="Purok 1, Barangay 10, Cagayan De Oro City, Misamis Oriental",
                     address_town_code=CDO, address_barangay="Barangay 10")
        assert cdo.status_code == 201, cdo.json()

    def test_a_barangay_name_with_a_comma(self, client, staff_token):
        """Codex verify round 2 (plan 002.1): "Agapito Yap, Sr." is one barangay (Sapang Dalaga)."""
        from app.services.barangay_gazetteer import town_barangays
        from app.api.routes.places import town_option_value
        from app.services.barangay_gazetteer import region_town
        code = "1004213000"
        names = town_barangays(code)[0]
        town = town_option_value(code, region_town(code))
        for name in ("Agapito Yap, Sr.", "Sixto Velez, Sr."):
            assert name in names
            for address in (f"Barangay {name}, {town}", f"Purok 2, Barangay {name}, {town}"):
                r = _issue(client, staff_token, address=address, address_town_code=code, address_barangay=name)
                assert r.status_code == 201, (address, r.json())
        # Still never a substring of another label.
        wrong = _issue(client, staff_token, address=f"Barangay Agapito Yap, Sr., {town}", address_town_code=code,
                       address_barangay="Sixto Velez, Sr.")
        assert wrong.status_code == 422

    def test_list_codes_are_request_only(self):
        # Never stored with a phone's QR form, never sent back in a scan (plan 002.1).
        from app.schemas.schemas import ApplicantCreate
        a = ApplicantCreate(**TestIssuance.SAMPLE_APPLICANT, place_of_birth_code="1", address_town_code="2",
                            address_barangay="Imbatug")
        assert a.place_of_birth_code == "1"
        dumped = a.model_dump()
        assert not {"place_of_birth_code", "address_town_code", "address_barangay"} & set(dumped)
        assert "place_of_birth_code" not in a.model_dump_json()


class TestPlan006Answers:
    """Plan 006: fillers are never answers, sex by sound, a town in several provinces lists them all."""

    @pytest.mark.parametrize("text,key,value", [
        ("Um, 60", "weight", "60"), ("Uh, 165", "height", "165"), ("Hmm, 20000", "gross_annual_income", "20000"),
    ])
    def test_a_filler_before_a_number_keeps_the_number(self, text, key, value):
        from app.services.answer_service import understand_answer
        result = understand_answer(key, text, spoken=True)
        assert result.ok is True and value in result.value

    @pytest.mark.parametrize("filler", ["Um.", "Uh", "Uhm", "Hmm", "Mm", "Ah", "Eh", "Er", "UM!", "um... uh", "Mm-hmm."])
    @pytest.mark.parametrize("key", ["surname", "given_name", "middle_name", "address_purok", "sex", "civil_status",
                                     "date", "occupation", "citizenship", "icr_no", "place_of_birth", "height"])
    @pytest.mark.parametrize("spoken", [True, False])
    def test_filler_only_is_an_empty_answer_for_every_field(self, key, filler, spoken):
        from app.services.answer_service import understand_answer
        result = understand_answer(key, filler, spoken=spoken)
        assert result.ok is False and result.reason == "empty"
        assert result.message.startswith("I didn't hear an answer. Please tell me your ")

    @pytest.mark.parametrize("text,key,value", [
        ("Um, Purok 2", "address_purok", "Purok 2"), ("Uh, male", "sex", "Male"), ("um... Juan", "given_name", "Juan"),
    ])
    def test_a_filler_inside_a_real_answer_is_removed(self, text, key, value):
        from app.services.answer_service import understand_answer
        result = understand_answer(key, text, spoken=True)
        assert result.ok is True and result.value == value

    @pytest.mark.parametrize("surname", ["Umali", "Uy", "Ermita", "Ahmad", "Hermano", "Mamon"])
    def test_surnames_that_look_like_fillers_survive(self, surname):
        from app.services.answer_service import understand_answer
        result = understand_answer("surname", surname)
        assert result.ok is True and result.value.lower() == surname.lower()

    @pytest.mark.parametrize("text,expected", [
        ("Meal", "Male"), ("Mehl.", "Male"), ("Mail", "Male"), ("Mel", "Male"), ("Meyl", "Male"), ("Mayl", "Male"),
        ("Nail", "Male"), ("Mile", "Male"), ("Male.", "Male"),
        ("Femail", "Female"), ("Feemale", "Female"), ("Pemale", "Female"), ("Fimale", "Female"),
        ("Bimale", "Female"), ("Female.", "Female"),
    ])
    def test_sex_by_sound(self, text, expected):
        from app.services.answer_service import understand_answer
        result = understand_answer("sex", text, spoken=True)
        assert result.ok is True and result.value == expected

    @pytest.mark.parametrize("text", ["I'm", "Email", "Animal", "Meal time", "Nails", "", "Fail", "Feel"])
    def test_sex_is_not_guessed(self, text):
        from app.services.answer_service import understand_answer
        result = understand_answer("sex", text, spoken=True)
        assert result.ok is False

    def test_a_town_in_several_provinces_lists_every_province(self):
        from app.services.answer_service import understand_answer
        result = understand_answer("place_of_birth_province", "San Jose", spoken=True)
        assert result.ok is False and result.reason == "ambiguous_town"
        assert len(result.candidates) > len(result.suggestions) == 3
        assert all(c["value"].startswith("San Jose") and c["province_psgc"] for c in result.candidates)

    def test_the_answer_route_returns_the_candidates(self, client, staff_token):
        resp = client.post("/api/extraction/answer", headers={"Authorization": f"Bearer {staff_token}"},
                           json={"field_key": "place_of_birth_province", "transcript": "San Jose", "spoken": True})
        assert resp.status_code == 200 and resp.json()["reason"] == "ambiguous_town"
        assert len(resp.json()["candidates"]) > 3

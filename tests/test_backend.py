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

    def test_get_record(self, client, staff_token):
        resp = client.get(
            f"/api/forms/records/{self.__class__.last_id}",
            headers={"Authorization": f"Bearer {staff_token}"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["ctc_number"] == self.__class__.last_ctc

    def test_list_records(self, client, staff_token):
        resp = client.get(
            "/api/forms/records",
            headers={"Authorization": f"Bearer {staff_token}"},
        )
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) >= 1

    def test_search_records(self, client, staff_token):
        resp = client.get(
            "/api/forms/records?search=Reyes",
            headers={"Authorization": f"Bearer {staff_token}"},
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
        assert client.get("/api/forms/records", headers=headers).status_code == 200
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

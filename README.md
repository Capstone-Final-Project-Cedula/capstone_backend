# Cedula Automated System — Backend API

Speech Recognition-Based System for Automated Community Tax Certificate (Cedula)
Data Entry and Form Generation for barangay-level government offices in the Philippines.

## Tech Stack

| Layer | Technology |
|---|---|
| Framework | FastAPI + Uvicorn |
| Database | SQLAlchemy + SQLite (dev) / MySQL (prod) |
| ASR | OpenAI Whisper (`base` model) |
| NLP/NER | DistilBERT (`dslim/distilbert-NER`) |
| PDF | ReportLab |
| QR | qrcode + pyzbar |
| TTS | pyttsx3 |
| Auth | JWT (python-jose + passlib) |

---

## Project Structure

```
cedula_system/
├── app/
│   ├── main.py                  # FastAPI app factory + startup
│   ├── core/
│   │   ├── config.py            # Settings from .env
│   │   ├── database.py          # SQLAlchemy engine + session
│   │   └── security.py         # JWT auth + password hashing
│   ├── models/
│   │   ├── applicant.py         # ApplicantRecord table
│   │   ├── issuance.py          # CTCIssuance table
│   │   ├── mobile_submission.py # MobileSubmission (QR) table
│   │   └── staff_user.py        # StaffUser table + RBAC
│   ├── schemas/
│   │   └── schemas.py           # Pydantic v2 request/response models
│   ├── services/
│   │   ├── speech_service.py    # Whisper ASR integration
│   │   ├── extraction_service.py# DistilBERT NER + Regex extraction
│   │   ├── tax_service.py       # BIR community tax computation
│   │   ├── pdf_service.py       # BIR Form 0016 PDF generation
│   │   ├── qr_service.py        # QR code generate + decode
│   │   └── tts_service.py       # pyttsx3 voice feedback
│   └── api/
│       └── routes/
│           ├── auth.py          # POST /api/auth/login, /register
│           ├── speech.py        # POST /api/speech/transcribe
│           ├── extraction.py    # POST /api/extraction/fields, /tax
│           ├── qr.py            # POST /api/qr/generate, /scan
│           └── forms.py         # POST /api/forms/issue, GET /records
├── tests/
│   └── test_backend.py          # pytest test suite (50+ tests)
├── requirements.txt
├── .env.example
└── README.md
```

---

## Quick Start

### 1. Clone and set up environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings (database, barangay name, etc.)
```

### 3. Run the server

```bash
# From the cedula_system/ directory
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open API docs

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

### 5. Default admin credentials

On first run, a default admin account is created:
- **Username:** `admin`
- **Password:** `admin1234`

> ⚠️ **Change this immediately in production!**

---

## API Endpoints Summary

### Authentication
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/auth/login` | Staff login → JWT token |
| POST | `/api/auth/register` | Create staff account (admin only) |
| GET | `/api/auth/me` | Current user profile |

### Speech Recognition
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/speech/transcribe` | Upload audio → Whisper transcript |
| POST | `/api/speech/verify` | TTS read-back of extracted fields |
| POST | `/api/speech/script` | Get verification script text |

### Field Extraction
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/extraction/fields` | Extract BIR Form 0016 fields from transcript |
| POST | `/api/extraction/tax` | Compute community tax |

### QR Code
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/qr/generate` | Generate QR from mobile form data |
| POST | `/api/qr/scan` | Claim QR by token (desktop) |
| POST | `/api/qr/decode-image` | Decode QR from uploaded image |

### Forms & Records
| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/forms/issue` | Full CTC issuance (save + PDF) |
| GET | `/api/forms/download/{ctc}` | Download Cedula PDF |
| GET | `/api/forms/records` | List all issuances |
| GET | `/api/forms/records/{id}` | Single issuance details |
| GET | `/api/forms/dashboard/stats` | Dashboard statistics |

---

## Running Tests

```bash
# Install test dependencies
pip install pytest httpx

# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=app --cov-report=term-missing
```

---

## Voice Processing Workflow (5 Steps)

```
1. AUDIO RECORDING
   Staff clicks "Record" → browser/desktop captures audio

2. WHISPER TRANSCRIPTION
   POST /api/speech/transcribe
   Audio → OpenAI Whisper → raw English text

3. FIELD EXTRACTION
   POST /api/extraction/fields
   Transcript → DistilBERT NER + Regex → BIR Form 0016 fields

4. TAX COMPUTATION
   Auto-computed from gross_annual_income using RA 7160 rules

5. VOICE FEEDBACK VERIFICATION
   POST /api/speech/verify
   System reads fields back to applicant via TTS for confirmation

   ↓ (confirmed)

6. ISSUE & GENERATE
   POST /api/forms/issue
   Save applicant + generate CTC number + generate PDF
```

## QR Workflow

```
[Citizen Android App]          [Desktop Barangay Staff]
      |                               |
  Fill BIR Form                       |
      |                               |
  POST /api/qr/generate               |
      |                               |
  Display QR (60-min expiry)          |
      |               ←──── Scan QR ──|
      |                               |
      |               POST /api/qr/scan
      |               ↓
      |        Form auto-populated
      |               ↓
      |        POST /api/forms/issue
      |               ↓
      |           PDF Generated
```

---

## Production Deployment

### MySQL Setup
```env
DATABASE_URL=mysql+pymysql://cedula_user:password@localhost:3306/cedula_db
```

### Environment Variables (production)
```env
DEBUG=False
SECRET_KEY=<random 64-char string>
JWT_SECRET_KEY=<random 64-char string>
BARANGAY_NAME=<your barangay name>
MUNICIPALITY=<your municipality>
PROVINCE=<your province>
```

### Run with Gunicorn (production)
```bash
gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

---

## Performance Targets
| Metric | Target | Test Method |
|---|---|---|
| Whisper WER | ≤ 15% | Compare vs ground truth transcripts |
| Field extraction accuracy | ≥ 90% | Compare vs actual client data |
| QR scan-to-form | ≤ 5 seconds | Stopwatch / API timing |
| SUS Score | ≥ 70 | ISO/IEC 25010 questionnaire |

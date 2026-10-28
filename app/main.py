"""
app/main.py
FastAPI application factory.
Registers all routers, middleware, startup/shutdown events.
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.concurrency import run_in_threadpool
from fastapi.exception_handlers import http_exception_handler
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.core.config import settings
from app.core.database import (
    DATABASE_KIND, create_tables, missing_eval_message, missing_migration_message,
)
from app.api.routes import auth, speech, extraction, qr, forms, citizen_auth, places
from app.api.routes import settings as settings_routes
from app.api.routes import sus, eval as eval_routes

# ─── Logging Setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Lifespan (startup / shutdown) ────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application startup: create DB tables and required directories.
    Application shutdown: cleanup.
    """
    logger.info("=== Cedula System Starting Up ===")

    # Configure PyTorch for low-memory container inference:
    # 1 thread prevents OpenMP from allocating thread pools for 64 host cores.
    # Disabling gradients stops PyTorch from tracking autograd graphs.
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_grad_enabled(False)
        logger.info("PyTorch configured: 1 thread, gradients disabled.")
    except Exception as exc:
        logger.warning(f"PyTorch thread constraint skipped: {exc}")

    # Which database this is (never the URL: it holds the password).
    logger.info(f"Database: {DATABASE_KIND}")

    # Stop here if the database lacks a migration: otherwise every save at the
    # interview station would fail mid-interview (plan 001.4).
    problem = missing_migration_message() or missing_eval_message()
    if problem:
        logger.error(problem)
        raise RuntimeError(problem)

    # Create DB tables (includes citizen_users now)
    create_tables()
    logger.info("Database tables verified/created.")

    # Create required directories
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
    os.makedirs(settings.PDF_OUTPUT_DIR, exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    logger.info("Required directories created.")

    # Seed a default admin user if the database is empty
    _seed_default_admin()

    # Load Whisper and DistilBERT in the background so the first interview answer is not slow.
    import threading
    if settings.WHISPER_PRELOAD:
        from app.services.speech_service import SpeechService
        threading.Thread(target=SpeechService.warm_up, name="whisper-warmup", daemon=True).start()
    if settings.NER_ENABLED:
        from app.services.answer_service import warm_up_ner
        threading.Thread(target=warm_up_ner, name="ner-warmup", daemon=True).start()

    logger.info(f"Barangay: {settings.BARANGAY_NAME}, {settings.MUNICIPALITY}, {settings.PROVINCE}")
    logger.info("=== Cedula System Ready ===")

    yield  # Application runs here

    logger.info("=== Cedula System Shutting Down ===")


def _seed_default_admin():
    """Create a default admin account if no staff users exist."""
    from app.core.database import SessionLocal
    from app.core.security import hash_password
    from app.models.staff_user import StaffUser, RoleEnum

    db = SessionLocal()
    try:
        count = db.query(StaffUser).count()
        if count == 0:
            default_admin = StaffUser(
                username="admin",
                hashed_password=hash_password("admin1234"),
                full_name="System Administrator",
                position="IT Administrator",
                role=RoleEnum.ADMIN,
            )
            db.add(default_admin)
            db.commit()
            logger.warning(
                "Default admin created: username='admin' password='admin1234' "
                "— CHANGE THIS IMMEDIATELY IN PRODUCTION!"
            )
    finally:
        db.close()


# ─── App Factory ───────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Cedula Automated System API",
        description=(
            "Speech Recognition-Based System for Automated Community Tax Certificate "
            "(Cedula) Data Entry and Form Generation. "
            "Developed for barangay-level government offices in the Philippines."
        ),
        version=settings.APP_VERSION,
        lifespan=lifespan,
        # The interactive API browser is for development only; with DEBUG off
        # (backend reachable from the internet through a tunnel) it is hidden.
        docs_url="/docs" if settings.DEBUG else None,
        redoc_url="/redoc" if settings.DEBUG else None,
        openapi_url="/openapi.json" if settings.DEBUG else None,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        # Production: only the web app's own address(es), from CORS_ORIGINS.
        # (The mobile app is not a browser, so CORS does not apply to it.)
        allow_origins=["*"] if settings.DEBUG else settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        # Let the browser read the total needed for the Records page buttons, and
        # the file name (CTC-..._COPY.pdf for a re-print) of a Records download.
        expose_headers=["X-Total-Count", "Content-Disposition", "X-Empty-Reason"],
    )

    # Failed logins and 401/403 answers are counted (time, route, kind; plan 003). Never blocks the answer.
    @app.exception_handler(StarletteHTTPException)
    async def count_denied_requests(request, exc):
        if exc.status_code in (401, 403):
            path = request.url.path
            kind = "login_failed" if "login" in path else ("unauthorized" if exc.status_code == 401 else "forbidden")
            from app.services import eval_service
            await run_in_threadpool(eval_service.record_security_event, path, kind)
        return await http_exception_handler(request, exc)

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(auth.router,         prefix="/api/auth",       tags=["Staff Authentication"])
    app.include_router(speech.router,       prefix="/api/speech",     tags=["Speech Recognition"])
    app.include_router(extraction.router,   prefix="/api/extraction", tags=["Field Extraction"])
    app.include_router(qr.router,           prefix="/api/qr",         tags=["QR Code"])
    app.include_router(forms.router,        prefix="/api/forms",      tags=["Forms & Records"])
    app.include_router(citizen_auth.router, prefix="/api/citizen",    tags=["Citizen Auth (Mobile)"])
    app.include_router(settings_routes.router, prefix="/api/settings", tags=["Settings"])
    app.include_router(sus.router,          prefix="/api/sus",        tags=["SUS Survey"])
    app.include_router(eval_routes.router,  prefix="/api/eval",        tags=["Evaluation mode"])
    app.include_router(places.router,       prefix="/api/places",     tags=["Place lists (pickers)"])

    # Generated PDFs are NOT served as public static files: they contain
    # citizens' personal data and have guessable names. Downloads go through
    # GET /api/forms/download/{ctc_number}, which requires a staff login.

    # ── Root Health Check ─────────────────────────────────────────────────────
    @app.get("/", tags=["Health"])
    def health_check():
        return {
            "status": "running",
            "system": "Cedula Automated System",
            "version": settings.APP_VERSION,
            "barangay": settings.BARANGAY_NAME,
        }

    @app.get("/api/health", tags=["Health"])
    def api_health():
        return {"status": "ok", "version": settings.APP_VERSION}

    return app


app = create_app()


# ─── Dev Entry Point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )

"""
app/main.py
FastAPI application factory.
Registers all routers, middleware, startup/shutdown events.
"""
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import settings
from app.core.database import create_tables
from app.api.routes import auth, speech, extraction, qr, forms, citizen_auth  # ← NEW

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
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
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
        # Let the browser read the total needed for the Records page buttons.
        expose_headers=["X-Total-Count"],
    )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(auth.router,         prefix="/api/auth",       tags=["Staff Authentication"])
    app.include_router(speech.router,       prefix="/api/speech",     tags=["Speech Recognition"])
    app.include_router(extraction.router,   prefix="/api/extraction", tags=["Field Extraction"])
    app.include_router(qr.router,           prefix="/api/qr",         tags=["QR Code"])
    app.include_router(forms.router,        prefix="/api/forms",      tags=["Forms & Records"])
    app.include_router(citizen_auth.router, prefix="/api/citizen",    tags=["Citizen Auth (Mobile)"])  # ← NEW

    # ── Static Files (serve generated PDFs for download) ─────────────────────
    os.makedirs(settings.PDF_OUTPUT_DIR, exist_ok=True)
    app.mount(
        "/pdfs",
        StaticFiles(directory=settings.PDF_OUTPUT_DIR),
        name="pdfs",
    )

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

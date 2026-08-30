"""
LEGALMET AI — FastAPI Application Entry Point
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
import sys
import io
import socket

# Ensure UTF-8 output on Windows console
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from backend.config import settings
from backend.database.database import init_db
from backend.api.routes.auth import router as auth_router
from backend.api.routes.inspections import router as inspections_router
from backend.api.routes.dashboard import router as dashboard_router
from backend.database.database import SessionLocal
from backend.database.models import User, UserRole
from backend.auth.auth import hash_password


def _seed_default_admin():
    """
    Create a default admin account if no users exist.
    Credentials are printed to stdout in development mode only.
    Change immediately after first login.
    """
    db = SessionLocal()
    try:
        if db.query(User).count() == 0:
            admin = User(
                username="admin",
                full_name="System Administrator",
                badge_number="ADMIN-001",
                department="Legal Metrology",
                role=UserRole.ADMIN,
                hashed_password=hash_password("Admin@1234!"),
                is_active=True,
            )
            inspector = User(
                username="inspector1",
                full_name="Inspector Demo",
                badge_number="INS-001",
                department="Legal Metrology - Field",
                role=UserRole.INSPECTOR,
                hashed_password=hash_password("Inspector@1234!"),
                is_active=True,
            )
            db.add(admin)
            db.add(inspector)
            db.commit()
            if settings.debug:
                print("\n" + "=" * 60)
                print("LEGALMET AI — Default accounts created (DEVELOPMENT ONLY)")
                print("  Admin:     admin / Admin@1234!")
                print("  Inspector: inspector1 / Inspector@1234!")
                print("  CHANGE THESE IN PRODUCTION.")
                print("=" * 60 + "\n")
    finally:
        db.close()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    init_db()
    _seed_default_admin()
    # Mount evidence image files as static files
    uploads_dir = os.path.abspath(settings.upload_dir)
    os.makedirs(uploads_dir, exist_ok=True)
    yield
    # Shutdown


app = FastAPI(
    title="LEGALMET AI",
    description=(
        "AI-Assisted Legal Metrology Inspection Platform — SIH Problem Statement SIH26034.\n\n"
        "**AI-Assisted Preliminary Assessment System.** "
        "Final statutory determination remains with the authorized officer."
    ),
    version="1.0.0-mvp",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth_router)
app.include_router(inspections_router)
app.include_router(dashboard_router)

# ── Static files (evidence crops & frontend) ───────────────────────────────────
_uploads_abs = os.path.abspath(settings.upload_dir)
if os.path.exists(_uploads_abs):
    app.mount("/uploads", StaticFiles(directory=_uploads_abs), name="uploads")

_frontend_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "frontend"))
_frontend_index = os.path.join(_frontend_dir, "index.html")
if os.path.exists(_frontend_dir):
    app.mount("/static", StaticFiles(directory=_frontend_dir), name="static")


@app.get("/")
def root():
    if os.path.exists(_frontend_index):
        return FileResponse(_frontend_index, media_type="text/html")
    return {
        "service": "LEGALMET AI",
        "subtitle": "AI-Assisted Legal Metrology Inspection Platform",
        "problem_statement": "SIH26034",
        "docs": "/docs",
        "disclaimer": (
            "AI-assisted preliminary assessment system. "
            "Final statutory determination remains with the authorized officer."
        ),
    }


@app.get("/api")
def api_info():
    return {
        "service": "LEGALMET AI API",
        "subtitle": "AI-Assisted Legal Metrology Inspection Platform",
        "problem_statement": "SIH26034",
        "docs": "/docs",
        "disclaimer": (
            "AI-assisted preliminary assessment system. "
            "Final statutory determination remains with the authorized officer."
        ),
    }


@app.get("/api/system/network")
def get_network_info():
    """Returns local network LAN IP addresses for mobile browser testing."""
    hostname = socket.gethostname()
    try:
        ip_list = socket.gethostbyname_ex(hostname)[2]
    except Exception:
        ip_list = []
    lan_ips = [ip for ip in ip_list if not ip.startswith("127.")]
    return {
        "hostname": hostname,
        "local_ips": lan_ips,
        "primary_url": f"http://{lan_ips[0]}:8000" if lan_ips else "http://localhost:8000",
        "all_urls": [f"http://{ip}:8000" for ip in lan_ips],
    }

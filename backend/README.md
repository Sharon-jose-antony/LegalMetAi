# LEGALMET AI — Backend Service

**AI-Assisted Legal Metrology Compliance Inspection Platform (SIH26034)**

The backend is built with **FastAPI**, **SQLAlchemy**, **OpenCV**, and **PaddleOCR / EasyOCR** to provide an automated, preliminary compliance screening pipeline for packaged commodities under the Legal Metrology (Packaged Commodities) Rules, 2011.

---

## 🏗 Architecture & Modules

```
backend/
├── main.py                     # FastAPI application entry point, CORS, static routes
├── config.py                   # Pydantic BaseSettings environment configuration
├── requirements.txt            # Python dependencies
├── .env.example                # Environment variables template
├── seed_demos.py               # Pre-seeded demo cases generator
│
├── api/                        # REST API Router Definitions
│   └── routes/
│       ├── auth.py             # User authentication (/api/auth/login, /api/auth/me)
│       ├── inspections.py      # Inspection lifecycle & AI pipeline orchestration
│       └── dashboard.py        # Executive enforcement statistics
│
├── auth/                       # Security, JWT tokens, bcrypt password hashing
├── database/                   # SQLite database engine, session factory & SQLAlchemy models
├── reports/                    # ReportLab PDF report generation
├── rules/                      # Consolidated Legal Metrology rules catalog (JSON)
├── schemas/                    # Pydantic v2 validation models & data contracts
└── services/                   # Core Computer Vision & AI Pipeline
    ├── image_quality.py        # Laplacian blur, contrast, glare, and resolution assessment
    ├── preprocessing.py        # Multi-variant OpenCV filters, CLAHE, adaptive thresholding
    ├── ocr.py                  # Pluggable OCR interface (PaddleOCR, EasyOCR, Mock)
    ├── declaration_extractor.py# Context-aware statutory declaration parser & regexes
    ├── rule_engine.py          # Category-aware deterministic Legal Metrology rule engine
    ├── evidence.py             # Declaration bounding box evidence cropper
    ├── confidence.py           # Multi-signal prototype confidence score synthesis
    └── pipeline.py             # Master inspection pipeline orchestrator
```

---

## 📦 Installation & Prerequisites

1. **Python 3.10+** (Tested on Python 3.10, 3.11, 3.12, 3.14).
2. Create and activate a Python virtual environment:
   ```bash
   python -m venv .venv
   # Windows (PowerShell):
   .\.venv\Scripts\Activate.ps1
   # Linux / macOS:
   source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r backend/requirements.txt
   ```

---

## ⚙ Environment Configuration

Copy `backend/.env.example` to `.env`:
```bash
cp backend/.env.example .env
```

Key environment variables:
- `DATABASE_URL`: SQLAlchemy connection string (default: `sqlite:///./legalmet_ai.db`).
- `SECRET_KEY`: JWT signing secret key (minimum 32 characters).
- `UPLOAD_DIR`: Directory for storing uploaded photos and evidence crops (default: `./data/uploads`).
- `REPORTS_DIR`: Directory for generated PDF reports (default: `./data/reports`).
- `OCR_PROVIDER`: `paddleocr` | `easyocr` | `mock` | `auto` (default: `auto`).

---

## 🚀 Running the Server

Start the FastAPI application with Uvicorn:
```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Inspector Web UI**: `http://localhost:8000/`
- **Interactive Swagger API Docs**: `http://localhost:8000/docs`
- **ReDoc API Reference**: `http://localhost:8000/redoc`

### Default Demo Accounts (Development Only)
- **Admin**: `admin` / `Admin@1234!`
- **Inspector**: `inspector1` / `Inspector@1234!`

---

## 🧪 Testing & Verification

Run the complete test suite (unit tests, accuracy suite, and benchmark tests):
```bash
python -m pytest tests/test_backend.py tests/test_accuracy_suite.py -v
```

Run the end-to-end integration verification script:
```bash
python tests/verify_e2e.py
```

---

## 📜 Statutory Legal Notice

LEGALMET AI provides an **AI-assisted preliminary compliance assessment**. Assessments are based on computer vision, multi-pass optical character recognition, and statutory pattern matching. Final legal determinations and physical enforcement actions remain the statutory responsibility of authorized Legal Metrology officers under the Legal Metrology Act, 2009.

# LEGALMET AI — AI-Assisted Legal Metrology Compliance Inspection Platform

[![SIH26034](https://img.shields.io/badge/SIH-SIH26034-blue.svg)](https://www.sih.gov.in/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI%200.110+-green.svg)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.14-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![OCR](https://img.shields.io/badge/OCR-PaddleOCR%20%2B%20EasyOCR-orange.svg)](https://github.com/PaddlePaddle/PaddleOCR)

**Smart India Hackathon (SIH26034)**  
**Organization**: Ministry of Consumer Affairs, Food & Public Distribution — Department of Consumer Affairs, Government of India.

---

## Table of Contents
1. [Problem Statement](#1-problem-statement)
2. [Solution Overview](#2-solution-overview)
3. [Core Architecture](#3-core-architecture)
4. [Technology Stack](#4-technology-stack)
5. [Repository Structure](#5-repository-structure)
6. [Prerequisites](#6-prerequisites)
7. [Backend Installation](#7-backend-installation)
8. [Frontend Installation](#8-frontend-installation)
9. [Environment Configuration](#9-environment-configuration)
10. [How to Run](#10-how-to-run)
11. [How to Test](#11-how-to-test)
12. [Phone Camera Testing & Field Inspection](#12-phone-camera-testing--field-inspection)
13. [OCR Pipeline & Multi-Pass Fusion](#13-ocr-pipeline--multi-pass-fusion)
14. [Statutory Declaration Extraction](#14-statutory-declaration-extraction)
15. [Deterministic Rule Engine](#15-deterministic-rule-engine)
16. [Evidence System & Bounding Box Cropping](#16-evidence-system--bounding-box-cropping)
17. [Evidentiary PDF Report Generation](#17-evidentiary-pdf-report-generation)
18. [Known Limitations](#18-known-limitations)
19. [Security Notes](#19-security-notes)
20. [SIH Demonstration Workflow](#20-sih-demonstration-workflow)

---

## 1. Problem Statement
Under the **Legal Metrology Act, 2009** read with the **Legal Metrology (Packaged Commodities) Rules, 2011 (PCR 2011)**, every packaged commodity in India must declare mandatory statutory consumer protections:
- Name and address of Manufacturer / Packer / Importer
- Common / Generic name of the commodity
- Net Quantity (standard metric units)
- Date / Month & Year of Manufacture / Packaging
- Maximum Retail Price (MRP inclusive of all taxes)
- Unit Sale Price (USP under GSR 779(E))
- Consumer Care contact coordinates (helpline, email, address)
- Best Before / Expiry date for perishable goods
- Country of Origin for imported items

Manual inspection across millions of retail SKUs and e-commerce listings is resource-intensive and error-prone. Non-compliant packages with missing declarations, obscured MRPs, or absent consumer care numbers often bypass verification.

---

## 2. Solution Overview
**LEGALMET AI** is an evidentiary preliminary screening and assessment system designed for Legal Metrology enforcement inspectors. It combines:
1. **Automated Image Quality Assessment**: Evaluates resolution, Laplacian variance blur, dynamic range, and specular glare.
2. **OpenCV Preprocessing Suite**: CLAHE, adaptive Gaussian thresholding, and unsharp masking for dot-matrix/faint print.
3. **Multi-Pass OCR Fusion Engine**: Integrates PaddleOCR (with directional angle classification `use_angle_cls=True`) and EasyOCR.
4. **Context-Aware Declaration Extractor**: Extracts and disambiguates statutory fields without misclassifying FSSAI License numbers, PIN codes, or Batch numbers as helpline phones.
5. **Category-Aware Deterministic Rule Engine**: Evaluates active rules against the official consolidated PCR 2011 catalog.
6. **Evidentiary PDF Generator**: Automatically formats preliminary inspection reports complete with cropped evidence thumbnails, legal citations, and inspector sign-off blocks.

> **Statutory Notice**: LEGALMET AI operates as an **AI-Assisted Preliminary Compliance Assessment System**. Final legal determination and enforcement actions remain the statutory prerogative of authorized Legal Metrology Officers under Section 15 of the Legal Metrology Act, 2009.

---

## 3. Core Architecture

```
                    [Packaged Commodity Image / Label]
                                   │
                                   ▼
                   ┌───────────────────────────────┐
                   │   Image Quality Assessment    │ (Resolution, Laplacian blur,
                   │   Service                     │  contrast, glare, aspect ratio)
                   └───────────────┬───────────────┘
                                   │
             ┌─────────────────────┴─────────────────────┐
             ▼                                           ▼
   [Quality Threshold Met]                     [Quality Insufficient]
             │                                           │
             ▼                                           ▼
   ┌───────────────────┐                       ┌───────────────────┐
   │ OpenCV Image      │ (Grayscale, CLAHE,    │ Recapture Prompt  │
   │ Preprocessing     │  Denoising, Sharpen,  │ & Route to Manual │
   │ & Enhancement     │  Adaptive Threshold)  │ Review Queue      │
   └─────────┬─────────┘                       └───────────────────┘
             │
             ▼
   ┌───────────────────────────────────────────┐
   │ OCRProvider Interface (Pluggable Engine)  │
   │ ├── Multi-pass: Original, Enhanced, Crops │
   │ └── Outputs: Normalized text, bboxes, conf│
   └─────────────────────┬─────────────────────┘
                         │
                         ▼
   ┌───────────────────────────────────────────┐
   │ Declaration Extractor & NLP Service       │
   │ (Context-Aware Regex + Lexical Patterns)  │
   │ Raw snippet preservation & field mapping  │
   └─────────────────────┬─────────────────────┘
                         │
                         ▼
   ┌───────────────────────────────────────────┐
   │ Product Category & Applicability Matrix   │ (Packaged Food, Cosmetics,
   │ └── Determines active mandatory rules     │  Household Commodity)
   └─────────────────────┬─────────────────────┘
                         │
                         ▼
   ┌───────────────────────────────────────────┐
   │ Deterministic Rule Compliance Engine      │ (Evaluates against official
   │ └── States: FOUND, NOT_FOUND,             │  consolidated rules catalog)
   │     NOT_APPLICABLE, UNCERTAIN,            │
   │     MANUAL_REVIEW                         │
   └─────────────────────┬─────────────────────┘
                         │
                         ▼
   ┌───────────────────────────────────────────┐
   │ Multi-Signal Prototype Confidence &       │ Image Quality + OCR Conf +
   │ Evidence Synthesis Matrix                 │ Extraction Conf + Applicability
   └─────────────────────┬─────────────────────┘
                         │
                         ▼
   ┌───────────────────────────────────────────┐
   │ Preliminary Assessment Outcome            │
   │ 🟢 PASS                                   │
   │ 🟡 MANUAL REVIEW                          │
   │ 🔴 POTENTIAL NON-COMPLIANCE               │
   └─────────────────────┬─────────────────────┘
                         │
             ┌───────────┴───────────┐
             ▼                       ▼
   ┌───────────────────┐   ┌───────────────────┐
   │ Inspector Review  │   │ AI-Assisted       │
   │ & Override Portal │   │ Preliminary       │
   │                   │   │ Inspection Report │
   │                   │   │ (PDF Generator)   │
   └───────────────────┘   └───────────────────┘
```

---

## 4. Technology Stack

- **Backend**: Python 3.10+, FastAPI, Uvicorn, SQLAlchemy, Pydantic v2.
- **Computer Vision & OCR**: OpenCV, NumPy, Pillow, PaddleOCR (PaddlePaddle), EasyOCR.
- **Reporting**: ReportLab PDF Generator.
- **Security & Auth**: JWT (OAuth2 Password Bearer), passlib (bcrypt).
- **Database**: SQLite (SQLAlchemy ORM).
- **Frontend**: Responsive Single-Page Application (HTML5, Vanilla CSS3, JavaScript ES6+), Google Fonts (*Plus Jakarta Sans*, *JetBrains Mono*).
- **Testing**: pytest, httpx.

---

## 5. Repository Structure

```
legalmet_ai/
│
├── frontend/                        # Inspector Single-Page Application
│   ├── index.html                   # Core frontend application & mobile UI
│   ├── package.json                 # Optional Node.js dev server scripts
│   ├── .env.example                 # Frontend environment template
│   └── README.md                    # Frontend documentation & mobile guide
│
├── backend/                         # FastAPI application backend
│   ├── main.py                      # Application entry point & static file routing
│   ├── config.py                    # Pydantic BaseSettings environment configuration
│   ├── requirements.txt             # Audited Python dependencies
│   ├── .env.example                 # Backend environment template
│   ├── seed_demos.py                # Pre-seeded demonstration cases
│   ├── api/                         # REST API routers (auth, inspections, dashboard)
│   ├── auth/                        # JWT authentication & role-based access control
│   ├── database/                    # SQLAlchemy database engine & data models
│   ├── services/                    # Core CV, OCR, extraction & rule engine services
│   ├── rules/                       # PCR 2011 legal rules catalog (JSON)
│   ├── reports/                     # ReportLab PDF report generation
│   ├── schemas/                     # Pydantic v2 data contracts
│   └── README.md                    # Backend architecture documentation
│
├── docs/                            # Documentation
│   ├── IMPLEMENTATION_PLAN.md       # Original blueprint & milestones
│   ├── architecture/
│   │   └── ARCHITECTURE.md          # Detailed engineering architecture
│   └── legal/
│       └── LEGAL_METROLOGY_RULES.md # Statutory Legal Metrology (PCR 2011) references
│
├── tests/                           # Automated test suites
│   ├── test_backend.py              # Unit tests for CV, OCR, rules & extractors
│   ├── test_accuracy_suite.py       # 25 packaged commodity benchmark regression suite
│   ├── test_mobile_workflow.py      # Field camera inspection simulation
│   └── verify_e2e.py                # 9-point end-to-end integration verification
│
├── test_data/                       # Evaluation & ground truth datasets
│   ├── README.md                    # Benchmark specification & instructions
│   └── samples/                     # Curated sample packaging images
│
├── data/                            # Runtime storage (.gitignored except .gitkeep)
│   ├── uploads/                     # Uploaded package photos & evidence crops
│   ├── reports/                     # Generated PDF reports
│   └── demo/                        # Pre-configured demonstration images
│
├── .gitignore                       # Production-grade Git ignore file
├── README.md                        # Master documentation
├── LICENSE                          # MIT License
├── start_backend.bat                # Windows portable startup script for Backend
└── start_frontend.bat               # Windows portable startup script for Frontend
```

---

## 6. Prerequisites

- **Python**: Version 3.10 or higher.
- **Node.js** (Optional): Version 18+ (only if running the frontend via a separate dev server).
- **Operating System**: Windows 10/11, Ubuntu 20.04+, or macOS.

---

## 7. Backend Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-org/legalmet_ai.git
   cd legalmet_ai
   ```

2. **Create and activate a virtual environment**:
   ```bash
   # Windows (Command Prompt / PowerShell)
   python -m venv .venv
   .\.venv\Scripts\activate

   # Linux / macOS
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Python dependencies**:
   ```bash
   pip install -r backend/requirements.txt
   ```

---

## 8. Frontend Installation

The frontend is a lightweight Single-Page Application served automatically by FastAPI. No compilation step is mandatory.

If you wish to run the frontend independently:
```bash
cd frontend
npm install
npm run dev
```

---

## 9. Environment Configuration

Copy the example environment configuration:
```bash
# Backend configuration
cp backend/.env.example .env

# Frontend configuration (if running standalone)
cp frontend/.env.example frontend/.env
```

Default `.env` configuration:
```env
APP_NAME="LEGALMET AI"
APP_ENV=development
DEBUG=true
SECRET_KEY=REPLACE_WITH_A_STRONG_RANDOM_SECRET_AT_LEAST_32_CHARS
DATABASE_URL=sqlite:///./legalmet_ai.db
UPLOAD_DIR=./data/uploads
REPORTS_DIR=./data/reports
OCR_PROVIDER=auto
```

---

## 10. How to Run

### Windows (Quick Start via Batch Scripts)
Double-click [`start_backend.bat`](start_backend.bat) or run from terminal:
```cmd
start_backend.bat
```

### Manual Command Line Startup
```bash
# Activate virtual environment
source .venv/bin/activate  # or .\.venv\Scripts\activate on Windows

# Start the FastAPI server
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

- **Inspector Web Portal**: [http://localhost:8000/](http://localhost:8000/)
- **Interactive Swagger API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc Reference**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

### Default Accounts (Development Only)
- **Admin**: `admin` / `Admin@1234!`
- **Inspector**: `inspector1` / `Inspector@1234!`

---

## 11. How to Test

Run the complete automated test suite (53 unit & benchmark tests):
```bash
python -m pytest tests/test_backend.py tests/test_accuracy_suite.py -v
```

Run the end-to-end integration and API verification:
```bash
python tests/verify_e2e.py
```

Run the mobile camera workflow simulation:
```bash
python tests/test_mobile_workflow.py
```

---

## 12. Phone Camera Testing & Field Inspection

LEGALMET AI supports direct mobile camera capture for field officers:
1. Connect your smartphone to the same Wi-Fi network as the laptop/server.
2. In the inspector sidebar, click the **Local Network IP** badge (or visit `/api/system/network`).
3. Open the displayed URL (e.g. `http://192.168.1.50:8000/`) in your smartphone browser.
4. Log in using `inspector1` / `Inspector@1234!`.
5. Tap **📸 Capture Product** — your smartphone's native camera opens automatically with high-resolution capture.
6. The system performs client-side orientation & compression, runs image quality checks, executes OCR, and displays the statutory compliance verdict in seconds.

---

## 13. OCR Pipeline & Multi-Pass Fusion

PaddleOCR and EasyOCR models are loaded via a pluggable provider interface ([`backend/services/ocr.py`](backend/services/ocr.py)):
- **Angle Classification (`use_angle_cls=True`)**: Identifies and corrects rotated or tilted text on curved bottles and flexible pouches.
- **Multi-Pass Execution**:
  1. *Pass 1*: Original crop.
  2. *Pass 2*: CLAHE contrast enhancement for faint/inkjet text.
  3. *Pass 3*: Adaptive Gaussian binarization for glossy/reflective foil.
- **Spatial Deduplication**: Overlapping detections are merged using IoU and sequence similarity, keeping the highest confidence reading and source coordinates.
- **Model Caching**: PaddleOCR/EasyOCR models download automatically on first run to the user cache (`~/.paddleocr/` and `~/.EasyOCR/`).

---

## 14. Statutory Declaration Extraction

The context-aware extractor ([`backend/services/declaration_extractor.py`](backend/services/declaration_extractor.py)) parses mandatory Legal Metrology fields:
- **MRP**: Recognizes ₹, Rs., INR symbols, and validates "inclusive of all taxes" under Rule 6(1)(e).
- **Net Quantity**: Standard metric units (g, kg, ml, L, units/pcs) under Rule 6(1)(c).
- **Unit Sale Price (USP)**: Mandated under GSR 779(E) for multi-piece or packages $> 1\text{kg}/\text{L}$.
- **Manufacturer / Packer**: Parses statutory prefixes (*Mfd by*, *Packed by*, *Mkt by*) and classifies Indian brand names and PIN codes.
- **Date & Expiry Coding**: Handles DD/MM/YYYY, MM/YY, "Best Before X Months from Packaging", and compact 6-digit batch dates.
- **Consumer Care Coordinates**: Captures toll-free 1800 numbers, emails, and postal addresses under Rule 6(1)(f).
- **Disambiguation Guard**: Strictly separates 14-digit FSSAI License numbers, 6-digit PIN codes, and Batch codes to avoid misclassifying them as consumer care contact numbers.

---

## 15. Deterministic Rule Engine

The rule engine ([`backend/services/rule_engine.py`](backend/services/rule_engine.py)) checks extracted values against [`backend/rules/legal_metrology_rules.json`](backend/rules/legal_metrology_rules.json):
- **Category Matrix**:
  - *Packaged Food*: Enforces Mfg, Net Qty, MRP, Mfg Date, Best Before, Consumer Care.
  - *Cosmetics / Personal Care*: Enforces Mfg, Net Qty, MRP, Mfg Date, Consumer Care (Best Before conditional).
  - *Household Commodity*: Enforces Mfg, Net Qty, MRP, Mfg Date, Consumer Care, Dimensions (Best Before N/A).
- **5 Evaluation States**: `FOUND`, `NOT_FOUND`, `NOT_APPLICABLE`, `UNCERTAIN`, `MANUAL_REVIEW`.
- **Verdict**:
  - 🟢 `PASS`: All mandatory statutory declarations validated.
  - 🟡 `MANUAL_REVIEW`: Low contrast, glare, or conditional declaration requiring physical verification.
  - 🔴 `POTENTIAL_NON_COMPLIANCE`: Missing mandatory declaration on acceptable quality image.

---

## 16. Evidence System & Bounding Box Cropping

For every detected statutory declaration, [`backend/services/evidence.py`](backend/services/evidence.py) extracts the physical pixel region from the package image:
- Saves crop thumbnails in `data/uploads/evidence/<INSPECTION_ID>/`.
- Associates exact bounding boxes `[ymin, xmin, ymax, xmax]` and source OCR tokens in the database.
- Renders an interactive evidence crop gallery in the inspector portal with lightbox zoom.

---

## 17. Evidentiary PDF Report Generation

Inspectors can generate and download an official preliminary inspection report ([`backend/reports/report_generator.py`](backend/reports/report_generator.py)):
- **Official Header**: Ministry of Consumer Affairs, Government of India.
- **Inspection Metadata**: ID format `LM-YYYY-XXXXXX`, timestamps, product category, inspector badge number.
- **Statutory Disclaimer**: Evidentiary preliminary assessment statement under Legal Metrology Act, 2009.
- **Summary Table**: Declarations, extracted values, confidence scores, and legal rule citations.
- **Visual Evidence Matrix**: Cropped image slices showing detected statutory markings.
- **Inspector Sign-Off**: Recorded statutory notes, override decision, and formal signature block.

---

## 18. Known Limitations

- **Physical Font Height Measurement**: Uncalibrated 2D smartphone photographs without physical calibration targets cannot guarantee absolute sub-millimeter measurements (e.g. Rule 9 letter heights). The system flags relative stroke legibility and routes borderline cases to `MANUAL_REVIEW`.
- **Obscured or Cylindrical 360° Packaging**: If declarations are distributed across multiple sides of a cylindrical tin or box, multiple photo captures may be required.

---

## 19. Security Notes

- **Password Hashing**: Passwords stored using `bcrypt` (12 rounds).
- **Token Expiry**: JWT access tokens configured with automated expiration.
- **Role-Based Access Control**: Strict segregation between `ADMIN` and `INSPECTOR` privileges.
- **Input Validation**: All uploaded files validated against allowed MIME types and file extensions.
- **No Secrets in Version Control**: `.gitignore` strictly excludes `.env` and local SQLite databases.

---

## 20. SIH Demonstration Workflow

Follow these steps for a live demonstration:

1. **Start the Application**:
   ```bash
   python -m uvicorn backend.main:app --port 8000 --reload
   ```
2. **Open Inspector Portal**: Navigate to [http://localhost:8000/](http://localhost:8000/) and click **👤 Inspector Demo** to sign in.
3. **Demo Scenario 1 (Compliant Baseline — `PASS`)**:
   - Inspect `LM-2026-DEMO01` (Packaged Food).
   - Observe 100% declaration extraction: MRP, Net Qty (80g), USP (₹0.06/g), Mfg Date, Best Before, Consumer Care helpline (1800-222-753).
   - Click **Download PDF Report** to view the generated evidentiary report.
4. **Demo Scenario 2 (Difficult Image / Glare — `MANUAL_REVIEW`)**:
   - Inspect `LM-2026-DEMO02` (Cosmetics / Face Serum).
   - Observe glare detection triggering multi-pass CLAHE enhancement and routing borderline text to `MANUAL_REVIEW`.
5. **Demo Scenario 3 (Non-Compliance — `POTENTIAL_NON_COMPLIANCE`)**:
   - Inspect `LM-2026-DEMO03` (Household Detergent).
   - Notice deliberate omission of Consumer Care contact under Rule 6(1)(f) and USP under GSR 779(E).
   - Open **Inspector Review**, record statutory observations, and submit the formal override.
6. **Live Field Test**:
   - Tap **📸 Capture Product** on a smartphone connected to local Wi-Fi to scan a live physical FMCG package.

---

## 📄 License
This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

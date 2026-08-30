# LEGALMET AI — Comprehensive Implementation Plan (SIH26034)
**AI-Assisted Legal Metrology Inspection Platform**

## 1. Statutory Context & Legal Foundation
- **SIH Problem Statement ID**: SIH26034
- **Title**: Software System to check compliance of Packaged Commodities under Legal Metrology (Packaged Commodities) Rules, 2011 by scanning products, images and labels.
- **Organization**: Ministry of Consumer Affairs, Food & Public Distribution — Department of Consumer Affairs.
- **Official Legal Framework**: Legal Metrology Act, 2009 read with Legal Metrology (Packaged Commodities) Rules, 2011 (consolidated with subsequent amendments including GSR 779(E) on Unit Sale Price, e-commerce declarations, and consumer care provisions).
- **Official References**:
  - Legal Metrology Act & Rules: https://consumeraffairs.gov.in/index.php/pages/legal-metrology-act
  - Legal Metrology Overview: https://consumeraffairs.gov.in/pages/legal-metrology-overview
- **Statutory Role & Position**: **AI-Assisted Preliminary Compliance Assessment System**. The system serves as an evidentiary and screening tool for enforcement inspectors; it does not replace statutory physical inspection, measurement verification, or final legal determination by an authorized officer.

---

## 2. Revised Target Architecture

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
   │ (Deterministic Regex + Lexical Patterns)  │
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

## 3. Product Category & Declaration Applicability Matrix

The system does not enforce a rigid static rule list. Instead, mandatory declarations are mapped conditionally based on the chosen Product Category.

| Declaration Field | Packaged Food | Cosmetics / Personal Care | Household Commodity | Applicability Condition / Legal Basis |
|---|:---:|:---:|:---:|---|
| **Name & Address of Manufacturer / Packer / Importer** | **Mandatory** | **Mandatory** | **Mandatory** | Rule 6(1)(a) — Required on all packages. |
| **Country of Origin** | **Conditional** | **Conditional** | **Conditional** | Rule 6(1)(aa) — Mandatory if product is imported or packaged abroad. |
| **Common or Generic Name** | **Mandatory** | **Mandatory** | **Mandatory** | Rule 6(1)(b) — Universal declaration for consumer identification. |
| **Net Quantity** | **Mandatory** | **Mandatory** | **Mandatory** | Rule 6(1)(c) & Second Schedule — Standard weight/volume/number units. |
| **Month & Year of Manufacture / Packing / Import** | **Mandatory** | **Mandatory** | **Mandatory** | Rule 6(1)(d) — Date coding format. |
| **Best Before / Expiry Date / Use By** | **Mandatory** | **Conditional** | **Not Applicable** | Mandated for perishables/food; conditional for cosmetics with limited shelf life; N/A for durable goods. |
| **Maximum Retail Price (MRP)** | **Mandatory** | **Mandatory** | **Mandatory** | Rule 6(1)(e) — "MRP ₹xx.xx (inclusive of all taxes)" or equivalent. |
| **Unit Sale Price (USP)** | **Conditional** | **Conditional** | **Conditional** | GSR 779(E) — Mandatory where net quantity is > 1 kg/litre or multi-piece package. |
| **Consumer Care Details** | **Mandatory** | **Mandatory** | **Mandatory** | Rule 6(1)(f) — Name, address, telephone/toll-free number, email of grievance officer. |
| **Dimensions / Size** | **Not Applicable** | **Not Applicable** | **Conditional** | Rule 6(1)(g) — Required if commodity is sold by size/dimension (e.g. foil, cloth, sheets). |

### Declaration Status Evaluation
- **FOUND**: Declaration detected, text extracted, validated against formatting rules.
- **NOT_FOUND**: Declaration missing on an applicable category where image quality was adequate.
- **NOT_APPLICABLE**: Declaration not legally required for the selected product category.
- **UNCERTAIN**: Text detected in candidate region but extraction confidence falls below threshold.
- **MANUAL_REVIEW**: Image blur, glare, partial occlusion, or relative text height prevents automated assessment.

---

## 4. Legal Metrology Rule Catalog Schema (`rules/legal_metrology_rules.json`)

```json
{
  "rule_id": "LM-PCR-2011-R6-1-E",
  "declaration": "MRP",
  "requirement": "Declaration of retail sale price (MRP) inclusive of all taxes in Indian Rupees.",
  "applicability": "All pre-packaged commodities intended for retail sale",
  "validation_logic": "mrp_pattern_match_and_tax_inclusion",
  "severity": "HIGH",
  "legal_reference": "Rule 6(1)(e), Legal Metrology (Packaged Commodities) Rules, 2011",
  "source_document": "Legal Metrology (Packaged Commodities) Rules, 2011 as amended",
  "source_date_version": "GSR 779(E) / Consolidated 2022",
  "official_source_url": "https://consumeraffairs.gov.in/index.php/pages/legal-metrology-act",
  "notes": "Must declare 'inclusive of all taxes'. Currency symbol ₹ or Rs. accepted."
}
```

---

## 5. OCR Architecture & Provider Abstraction

```python
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import numpy as np

class OCRResult:
    text: str
    confidence: float
    bbox: List[int] # [ymin, xmin, ymax, xmax]
    polygon: List[List[int]]
    source_pass: str # "original", "enhanced_clahe", "high_contrast_bin"

class OCRProvider(ABC):
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the underlying OCR engine."""
        pass

    @abstractmethod
    def extract_text(self, image_np: np.ndarray, pass_type: str = "original") -> List[OCRResult]:
        """Perform OCR on the supplied image array."""
        pass
```
- **Primary Provider**: EasyOCR / PaddleOCR Provider.
- **Fallback / Mock Provider**: Robust deterministic regex test harness for offline unit testing and predictable integration validation.
- **Multi-Pass Strategy**:
  1. *Pass 1 (Direct)*: Original image crop.
  2. *Pass 2 (CLAHE Contrast)*: Local contrast normalized image for faint inkjet/dot-matrix print.
  3. *Pass 3 (Otsu / Adaptive Threshold)*: Binarized image for glossy, glare-affected surfaces.

---

## 6. Text Readability & Relative Text-Size Analysis (Without Uncalibrated Statutory Claims)
- The system **does not** claim physical millimeter measurements (e.g. 1.5mm, 3.0mm height under Rule 9) from uncalibrated consumer camera photos without physical fiducial reference standards.
- Instead, the system calculates:
  - **Estimated Character Height (pixels)** relative to image height.
  - **Readability & Contrast Index** (Foreground/background luminance delta).
  - **Legibility Assessment**: If text height is below the optical resolution limit (~12px stroke), flag as `READABILITY_CONCERN` and route to `MANUAL_REVIEW`.

---

## 7. Prototype Confidence Scoring Model
- To maintain scientific integrity, confidence is explicitly labeled as **"Prototype Confidence Score"** and calculated as:
  $$C_{\text{prototype}} = w_q \cdot S_{\text{quality}} + w_o \cdot S_{\text{ocr}} + w_e \cdot S_{\text{extraction}} + w_a \cdot S_{\text{applicability}}$$
  where:
  - $S_{\text{quality}}$: Normalized Laplacian blur + contrast score ([0, 1]).
  - $S_{\text{ocr}}$: Mean OCR confidence across detected declaration tokens ([0, 1]).
  - $S_{\text{extraction}}$: Deterministic regex/pattern match completeness ([0, 1]).
  - $S_{\text{applicability}}$: Ratio of conclusively resolved declarations ([0, 1]).
  - Default weights: $w_q = 0.20, w_o = 0.35, w_e = 0.35, w_a = 0.10$.

---

## 8. Controlled Demo Scenarios
1. **DEMO 1 — Controlled Baseline Sample (High Compliance)**:
   - Category: *Packaged Food*.
   - Input: High-resolution clear package image with all mandatory declarations (Mfg, Net Qty, MRP ₹120 incl taxes, Mfg Date, Best Before, Consumer Care email/phone, Generic Name).
   - Expected Output: `PASS` (Prototype Confidence > 0.88).
2. **DEMO 2 — Difficult Image Sample (Low Contrast / Glare / Faint Text)**:
   - Category: *Cosmetics / Personal Care*.
   - Input: Curved cosmetic bottle with reflective foil glare and faint dot-matrix batch/MRP code.
   - Expected Output: Multi-pass enhancement triggers; Consumer Care region flagged as `UNCERTAIN` / `MANUAL_REVIEW` due to glare; Overall outcome: `MANUAL_REVIEW`.
3. **DEMO 3 — Controlled Synthetic / Deliberately Altered Test Sample**:
   - Clearly labeled in UI and report as: **"CONTROLLED DEMO TEST CASE"**.
   - Category: *Household Commodity*.
   - Input: Packaged detergent/cleaner where the mandatory Consumer Care phone number and Unit Sale Price (USP) have been deliberately omitted.
   - Expected Output: `POTENTIAL_NON_COMPLIANCE` (identifies missing Consumer Care contact under Rule 6(1)(f) and missing USP under GSR 779(E)).

---

## 9. Implementation Milestones

- **Milestone 1: Project Foundation & Environment Setup**
  - Scaffold `backend/` and `frontend/` directory structures.
  - Setup FastAPI app, health endpoints, CORS, Pydantic schemas.
  - Setup consolidated `rules/legal_metrology_rules.json`.
- **Milestone 2: Image Quality Assessment & Preprocessing Service**
  - Implement `services/image_quality.py` (Laplacian blur, contrast, glare, resolution).
  - Implement OpenCV enhancements (CLAHE, adaptive binarization, unsharp masking).
- **Milestone 3: Pluggable OCR Provider & Multi-Pass Execution**
  - Implement `OCRProvider` abstract class and concrete provider.
  - Multi-pass OCR pipeline returning normalized bounding boxes and confidences.
- **Milestone 4: Applicability-Aware Declaration Extractor (NLP / Regex)**
  - Extraction parsers for MRP, Net Qty, USP, Mfg/Packer/Importer, Generic Name, Country of Origin, Mfg/Expiry Dates, Consumer Care coordinates.
- **Milestone 5: Deterministic Legal Metrology Rule Engine**
  - Category-aware rule validation against official consolidated rules catalog.
  - Output mapping to `FOUND`, `NOT_FOUND`, `NOT_APPLICABLE`, `UNCERTAIN`, `MANUAL_REVIEW`.
- **Milestone 6: Evidence Extraction, Cropping & Confidence Synthesis**
  - Generate crop bounding box images for every detected declaration.
  - Compute weighted Prototype Confidence Score.
- **Milestone 7: Inspector Portal & Manual Review Override Service**
  - Store inspector decision (`CONFIRMED`, `OVERRIDDEN`, `REJECTED`), statutory notes, and audit timestamps.
- **Milestone 8: AI-Assisted Preliminary Inspection Report (ReportLab PDF)**
  - Generate official PDF with product image, extracted declaration table, evidence crops, legal references, inspector notes, and statutory disclaimer.
- **Milestone 9: SQLite Database Persistence & Audit History**
  - Schema for inspections, declarations, evidence, manual reviews, and users.
  - Inspection ID generation format: `LM-2026-XXXXXX`.
- **Milestone 10: Executive Enforcement Dashboard & Search**
  - Aggregated inspection metrics, category breakdowns, pass/review/non-compliance rates.
  - Multi-parameter search (Inspection ID, Brand, Date, Category, Status).
- **Milestone 11: Frontend UI Implementation (Flutter / Web Material 3)**
  - Build all core screens adhering to Stitch design tokens and responsive layouts.
- **Milestone 12: End-to-End Testing, Demo Suite & Documentation**
  - Automated pytest test suite for all services.
  - Three controlled demo scenario runners.
  - Complete technical documentation in `docs/`.

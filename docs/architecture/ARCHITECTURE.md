# LEGALMET AI — System Architecture & Technical Specifications

**SIH Problem Statement ID**: SIH26034  
**Title**: Software System to check compliance of Packaged Commodities under Legal Metrology (Packaged Commodities) Rules, 2011 by scanning products, images and labels.  
**Organization**: Ministry of Consumer Affairs, Food & Public Distribution — Department of Consumer Affairs.

---

## 1. End-to-End System Pipeline

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

## 2. Core Service Specifications

### A. Image Quality Assessment (`backend/services/image_quality.py`)
- **Resolution Check**: Evaluates if dimensions satisfy the optical stroke legibility minimum (default $\ge 400\text{px}$).
- **Laplacian Variance Blur Metric**:
  $$\text{Var}(\nabla^2 I) = \frac{1}{N}\sum (L(x,y) - \bar{L})^2$$
  Flags motion blur or camera defocus when variance falls below threshold ($80.0$).
- **Dynamic Range & Contrast**: Evaluates luminance standard deviation across grayscale histogram.
- **Specular Glare Detection**: Detects overexposed saturation spots from glossy laminated foil packaging.

### B. Image Preprocessing & Multi-Variant Fusion (`backend/services/preprocessing.py`)
- **Intelligent Upscaling**: Applies bilinear and Lanczos super-resolution scaling for low-resolution label crops, maintaining an inverse transform mapping back to original coordinates.
- **Contrast Limited Adaptive Histogram Equalization (CLAHE)**: Enhances faint dot-matrix printing and inkjet batch codes without amplifying background noise.
- **Adaptive Gaussian Thresholding**: Binarizes text regions under non-uniform illumination and reflective foil glare.
- **Unsharp Mask Sharpening**: Accentuates character edges.

### C. Multi-Pass OCR Architecture (`backend/services/ocr.py`)
- **Provider Abstraction**:
  - `PaddleOCRProvider`: DB/DB++ text detection, angle classification (`use_angle_cls=True`), CRNN text recognition.
  - `EasyOCRProvider`: CRAFT text detector with ResNet recognizer.
  - `MockOCRProvider`: Deterministic regression testing provider.
- **Multi-Pass Fusion**: Runs OCR passes across original and enhanced image variants.
- **Spatial & String-Similarity Deduplication**: Uses Intersection over Union (IoU), coverage heuristics, and Levenshtein/SequenceMatcher string similarity to fuse overlapping token detections while preserving the highest confidence detection and its bounding box.

### D. Context-Aware Declaration Extractor (`backend/services/declaration_extractor.py`)
- Extracts statutory declaration fields:
  1. `MRP` (Maximum Retail Price with tax inclusion check under Rule 6(1)(e))
  2. `NET_QUANTITY` (Metric weights/volumes/units under Rule 6(1)(c))
  3. `UNIT_SALE_PRICE` (USP under GSR 779(E))
  4. `MANUFACTURER_PACKER_IMPORTER` (Statutory prefix parsing + brand matching under Rule 6(1)(a))
  5. `COMMON_GENERIC_NAME` (Rule 6(1)(b))
  6. `MONTH_YEAR_OF_MANUFACTURE` (Rule 6(1)(d))
  7. `BEST_BEFORE_EXPIRY_DATE` (Rule 6(1)(d))
  8. `CONSUMER_CARE_DETAILS` (Rule 6(1)(f))
  9. `LICENSE_NUMBER` (FSSAI Lic. No.)
  10. `DIMENSIONS` (Rule 6(1)(g))
  11. `COUNTRY_OF_ORIGIN` (Rule 6(1)(aa))
- **Identifier Disambiguation**: Excludes FSSAI License numbers (14-digit), Postal PIN codes (6-digit), batch codes, and barcodes from being misclassified as consumer care phone numbers.

### E. Deterministic Rule Engine (`backend/services/rule_engine.py`)
- Maps extracted declarations against official consolidated requirements loaded from `rules/legal_metrology_rules.json`.
- Evaluates compliance across product categories (*Packaged Food*, *Cosmetics*, *Household Commodities*).
- Yields declaration states: `FOUND`, `NOT_FOUND`, `NOT_APPLICABLE`, `UNCERTAIN`, `MANUAL_REVIEW`.
- Aggregates overall status:
  - `PASS`: All mandatory rules satisfied.
  - `MANUAL_REVIEW`: Quality issues, uncertain readings, or conditional reviews required.
  - `POTENTIAL_NON_COMPLIANCE`: Missing mandatory declaration on acceptable quality image.

### F. Multi-Signal Prototype Confidence Score (`backend/services/confidence.py`)
Calculates a weighted, transparent confidence metric:
$$C_{\text{prototype}} = w_q \cdot S_{\text{quality}} + w_o \cdot S_{\text{ocr}} + w_e \cdot S_{\text{extraction}} + w_a \cdot S_{\text{applicability}}$$
- $w_q = 0.20$ (Image Quality Score)
- $w_o = 0.35$ (Mean OCR Token Confidence)
- $w_e = 0.35$ (Extraction Completeness Score)
- $w_a = 0.10$ (Applicability Resolution Ratio)

### G. Evidentiary PDF Report Generator (`backend/reports/report_generator.py`)
- Generates official legal metrology preliminary inspection reports in PDF format using ReportLab.
- Contains: Inspection ID (`LM-YYYY-XXXXXX`), inspection metadata, statutory disclaimer, product photo, extracted declarations table, evidence slice thumbnails, rule evaluation findings, inspector override section, and signature block.

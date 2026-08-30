# LEGALMET AI — Test Data & Benchmark Evaluation Guide

This directory contains ground truth datasets, test image formats, and instructions for running accuracy and regression benchmarks for the **LEGALMET AI** compliance platform.

---

## 📁 Directory Organization

```
test_data/
├── README.md                  # This guide
└── samples/                   # Curated public domain sample packaging images
    ├── demo1_food_compliant.png
    ├── demo2_cosmetic_glare.png
    └── demo3_household_non_compliant.png
```

---

## 🎯 25-Product Ground Truth Benchmark Suite

The automated benchmark suite in [`tests/test_accuracy_suite.py`](../tests/test_accuracy_suite.py) evaluates 25 real-world Indian packaged commodities across FMCG categories:

### Ground Truth Data Model (`ProductGroundTruth`)
Each ground truth test case is defined with:
- `product_id`: Unique benchmark identifier (e.g. `PROD-01`).
- `product_name`: Full commercial product name.
- `brand_name`: Brand manufacturer.
- `category`: `PACKAGED_FOOD` | `COSMETICS` | `HOUSEHOLD_COMMODITY`.
- `is_imported`: Boolean indicating whether country of origin is mandatory.
- `ocr_tokens`: Real-world OCR token stream including low-resolution dot-matrix artifacts.
- `full_text`: Raw concatenated OCR output text.
- `expected_declarations`: Dictionary mapping statutory declaration fields to expected values.

---

## 🔬 How to Run the Accuracy Benchmark

Execute the regression and accuracy benchmark suite using pytest:

```bash
python -m pytest tests/test_accuracy_suite.py -v
```

This verifies:
1. **Field-Level Extraction Accuracy**: Evaluates MRP, Net Quantity, Unit Sale Price, Manufacturer, Date coding, Expiry, and Consumer Care extraction against ground truth.
2. **Identifier Disambiguation**: Ensures 14-digit FSSAI numbers, 6-digit PIN codes, and Batch codes are never misclassified as consumer care phone numbers.
3. **Statutory Separation**: Confirms OCR detection quality issues route to `MANUAL_REVIEW` rather than false `POTENTIAL_NON_COMPLIANCE`.

---

## 📸 Adding New Test Images

When adding real test photos:
1. Place image files (JPG, PNG, or WEBP) in `test_data/samples/`.
2. Ensure image resolution is $\ge 400\text{px}\times 400\text{px}$ to satisfy stroke legibility thresholds.
3. Define the corresponding ground truth entry in `tests/test_accuracy_suite.py`.

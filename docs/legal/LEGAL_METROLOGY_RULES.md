# LEGALMET AI — Statutory Legal Metrology References & Compliance Rules

**SIH Problem Statement**: SIH26034  
**Primary Statutory Authority**: Legal Metrology Act, 2009 read with Legal Metrology (Packaged Commodities) Rules, 2011 as amended.  
**Administrative Ministry**: Ministry of Consumer Affairs, Food & Public Distribution — Department of Consumer Affairs, Government of India.

---

## 1. Statutory Foundations & Legal Overview

Under the **Legal Metrology (Packaged Commodities) Rules, 2011 (PCR 2011)**, every pre-packaged commodity intended for retail sale in India must bear mandatory declarations on the principal display panel or package exterior.

### Key Official Sources
- [Legal Metrology Act, 2009](https://consumeraffairs.gov.in/index.php/pages/legal-metrology-act)
- [Department of Consumer Affairs — Legal Metrology Overview](https://consumeraffairs.gov.in/pages/legal-metrology-overview)
- [Consolidated Packaged Commodities Rules & Amendments](https://consumeraffairs.gov.in/index.php/pages/legal-metrology-act)

---

## 2. Mandatory Declaration Requirements under Rule 6(1)

| Rule Clause | Declaration Field | Statutory Requirement | Applicability |
|---|---|---|---|
| **Rule 6(1)(a)** | **Manufacturer / Packer / Importer** | Name and complete address of the manufacturer, or where manufacturer is not packer, name and address of packer/importer. | Mandatory for all categories. |
| **Rule 6(1)(aa)** | **Country of Origin** | Name of the country of origin or manufacture or assembly for imported packages. | Conditional (Mandatory if imported). |
| **Rule 6(1)(b)** | **Common / Generic Name** | Common or generic name of the commodity contained in the package. | Mandatory for all categories. |
| **Rule 6(1)(c)** | **Net Quantity** | Net quantity in terms of standard unit of weight, measure, or number (g, kg, ml, L, units). | Mandatory for all categories. |
| **Rule 6(1)(d)** | **Month & Year of Manufacture** | Month and year in which the commodity is manufactured, pre-packed, or imported. | Mandatory for all categories. |
| **Rule 6(1)(d)** | **Best Before / Expiry Date** | Best before or use by date for commodities liable to perish or lose efficacy. | Mandatory for Food; Conditional for Cosmetics; N/A for durable Household items. |
| **Rule 6(1)(e)** | **Maximum Retail Price (MRP)** | Retail sale price expressed as "MRP ₹xx.xx (inclusive of all taxes)" or equivalent. | Mandatory for all retail packaged goods. |
| **GSR 779(E)** | **Unit Sale Price (USP)** | Unit Sale Price in terms of Rupees per g, kg, ml, L, or piece for packages $> 1\text{kg}/\text{L}$ or multi-piece packs. | Conditional (Mandatory where applicable under GSR 779(E)). |
| **Rule 6(1)(f)** | **Consumer Care Details** | Name, address, telephone number / toll-free helpline, and email address of person/cell for consumer grievances. | Mandatory for all categories. |
| **Rule 6(1)(g)** | **Dimensions / Size** | Dimensions of the commodity where sold by size/area (e.g. foils, sheets, cloth). | Conditional (Household / Paper goods sold by size). |

---

## 3. Product Category Applicability Matrix

LEGALMET AI implements dynamic applicability filtering to avoid false non-compliance flags:

```
Category: PACKAGED_FOOD
├── Mandatory: Manufacturer/Packer, Generic Name, Net Quantity, Mfg Date, Best Before, MRP, Consumer Care
└── Conditional: Country of Origin (if imported), Unit Sale Price (if >1kg/L)

Category: COSMETICS / PERSONAL CARE
├── Mandatory: Manufacturer/Packer, Generic Name, Net Quantity, Mfg Date, MRP, Consumer Care
└── Conditional: Best Before (shelf life dependent), Country of Origin, Unit Sale Price

Category: HOUSEHOLD_COMMODITY
├── Mandatory: Manufacturer/Packer, Generic Name, Net Quantity, Mfg Date, MRP, Consumer Care
├── Conditional: Dimensions / Size, Unit Sale Price, Country of Origin
└── Not Applicable: Best Before / Expiry Date
```

---

## 4. Rule Catalog Schema (`backend/rules/legal_metrology_rules.json`)

Each rule record maintains full statutory provenance:
```json
{
  "rule_id": "LM-PCR-2011-R6-1-E",
  "declaration": "MRP",
  "requirement": "Declaration of retail sale price (MRP) inclusive of all taxes in Indian Rupees.",
  "applicability": {
    "all_categories": true,
    "packaged_food": "MANDATORY",
    "cosmetics": "MANDATORY",
    "household_commodity": "MANDATORY"
  },
  "validation_logic": "mrp_pattern_match_and_tax_inclusion",
  "severity": "HIGH",
  "legal_reference": "Rule 6(1)(e), Legal Metrology (Packaged Commodities) Rules, 2011",
  "source_document": "Legal Metrology (Packaged Commodities) Rules, 2011 as amended",
  "source_date_version": "GSR 779(E) / Consolidated 2022",
  "official_source_url": "https://consumeraffairs.gov.in/index.php/pages/legal-metrology-act"
}
```

---

## 5. Evidentiary Disclaimer & Statutory Position

> **Statutory Notice**: LEGALMET AI operates as an **AI-Assisted Preliminary Compliance Assessment System**. The software serves as an evidentiary aid and screening tool for enforcement officers; it does not replace statutory physical inspection, physical weighing/measurement verification, or final legal determinations by an authorized Legal Metrology Inspector under Section 15 of the Legal Metrology Act, 2009.

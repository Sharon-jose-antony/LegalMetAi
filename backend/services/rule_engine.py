"""
LEGALMET AI — Deterministic Legal Metrology Rule Engine
Evaluates extracted declarations against the official rules catalog.

Design:
- Rules are loaded from rules/legal_metrology_rules.json at startup.
- Applicability is product-category-aware.
- OCR failure != legal violation.
- Insufficient evidence routes to MANUAL_REVIEW, not POTENTIAL_NON_COMPLIANCE.
"""
from __future__ import annotations
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any

from backend.services.declaration_extractor import ExtractedDeclaration
from backend.database.models import ProductCategory, DeclarationStatus


RULES_PATH = os.path.join(os.path.dirname(__file__), "..", "rules", "legal_metrology_rules.json")


@dataclass
class RuleEvalResult:
    rule_id: str
    declaration_field: str
    status: DeclarationStatus
    severity: str
    legal_reference: str
    official_source_url: str
    details: str


def _load_rules() -> List[dict]:
    path = os.path.abspath(RULES_PATH)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# Category string → applicability key in rules JSON
_CATEGORY_KEY: Dict[ProductCategory, str] = {
    ProductCategory.PACKAGED_FOOD: "packaged_food",
    ProductCategory.COSMETICS: "cosmetics",
    ProductCategory.HOUSEHOLD_COMMODITY: "household_commodity",
}


def _get_applicability(rule: dict, category: ProductCategory) -> str:
    """
    Returns the applicability status for a given rule + product category.
    Possible values: MANDATORY, CONDITIONAL, NOT_APPLICABLE
    """
    applicability_block = rule.get("applicability", {})
    if applicability_block.get("all_categories", False):
        return "MANDATORY"
    cat_key = _CATEGORY_KEY.get(category, "")
    return applicability_block.get(cat_key, "NOT_APPLICABLE").upper()


def evaluate_rules(
    extracted_declarations: List[ExtractedDeclaration],
    category: ProductCategory,
    quality_score: float,
    is_imported: bool = False,
) -> List[RuleEvalResult]:
    """
    Evaluate all applicable rules for the given product category and
    extracted declarations.

    Logic per declaration:
    - NOT_APPLICABLE applicability → skip (mark NOT_APPLICABLE)
    - CONDITIONAL → evaluate only if heuristics suggest applicability
    - MANDATORY:
        - Declaration extracted with confidence >= threshold → FOUND
        - Declaration extracted with low confidence → UNCERTAIN → MANUAL_REVIEW
        - Declaration not extracted, image quality POOR/UNUSABLE → MANUAL_REVIEW
        - Declaration not extracted, image quality ACCEPTABLE/GOOD → POTENTIAL_NON_COMPLIANCE candidate
    """
    rules = _load_rules()

    # Build lookup: field name → ExtractedDeclaration
    decl_map: Dict[str, ExtractedDeclaration] = {}
    for decl in extracted_declarations:
        decl_map[decl.field] = decl

    results: List[RuleEvalResult] = []

    for rule in rules:
        rule_id = rule["rule_id"]
        decl_field = rule["declaration"]
        applicability = _get_applicability(rule, category)

        # ── NOT_APPLICABLE ────────────────────────────────────────────────────
        if applicability == "NOT_APPLICABLE":
            results.append(RuleEvalResult(
                rule_id=rule_id,
                declaration_field=decl_field,
                status=DeclarationStatus.NOT_APPLICABLE,
                severity=rule.get("severity", "INFORMATIONAL"),
                legal_reference=rule.get("legal_reference", ""),
                official_source_url=rule.get("official_source_url", ""),
                details="Declaration not applicable to selected product category.",
            ))
            continue

        # ── CONDITIONAL: Country of Origin ────────────────────────────────────
        if applicability == "CONDITIONAL" and decl_field == "COUNTRY_OF_ORIGIN":
            if not is_imported:
                results.append(RuleEvalResult(
                    rule_id=rule_id,
                    declaration_field=decl_field,
                    status=DeclarationStatus.NOT_APPLICABLE,
                    severity=rule.get("severity", "INFORMATIONAL"),
                    legal_reference=rule.get("legal_reference", ""),
                    official_source_url=rule.get("official_source_url", ""),
                    details="Product is not flagged as imported; country of origin declaration is not required.",
                ))
                continue

        # ── Look up extracted declaration ─────────────────────────────────────
        decl = decl_map.get(decl_field)

        # ── Field extracted and confidence is meaningful ──────────────────────
        if decl and decl.extracted_value and decl.extraction_confidence >= 0.65:
            status = DeclarationStatus.FOUND
            details = f"Extracted value: '{decl.extracted_value}'. Extraction confidence: {decl.extraction_confidence:.2f}."

        elif decl and decl.extracted_value and decl.extraction_confidence >= 0.30:
            # Low confidence but something was found — uncertain
            status = DeclarationStatus.UNCERTAIN
            details = (
                f"Partial extraction: '{decl.extracted_value}'. "
                f"Low extraction confidence ({decl.extraction_confidence:.2f}). "
                "Manual verification recommended."
            )

        elif decl and decl.extraction_confidence < 0.30:
            # Extraction failed or near-zero confidence
            if quality_score < 0.60:
                status = DeclarationStatus.MANUAL_REVIEW
                details = (
                    "Declaration could not be extracted. Image quality is insufficient "
                    "for automated assessment (quality score: "
                    f"{quality_score:.2f}). Manual inspection required."
                )
            else:
                # Good image, nothing found → potential issue (not confirmed violation)
                if applicability == "MANDATORY":
                    status = DeclarationStatus.NOT_FOUND
                    details = (
                        "Declaration not detected on image of acceptable quality. "
                        "This may indicate the declaration is absent or placed in a region "
                        "not captured in this image. Manual verification recommended."
                    )
                else:
                    status = DeclarationStatus.MANUAL_REVIEW
                    details = "Conditional declaration not found; manual review required to determine applicability."

        else:
            # No extractor result at all
            if quality_score < 0.60:
                status = DeclarationStatus.MANUAL_REVIEW
                details = (
                    "Declaration could not be detected. Image quality / glare is insufficient "
                    "for automated assessment (quality score: "
                    f"{quality_score:.2f}). Manual inspection required."
                )
            elif applicability == "MANDATORY":
                status = DeclarationStatus.NOT_FOUND
                details = (
                    "Mandatory declaration not detected. Image quality is adequate. "
                    "Potential absence of declaration requires manual verification."
                )
            else:
                status = DeclarationStatus.MANUAL_REVIEW
                details = "Conditional declaration not found; manual review required."

        results.append(RuleEvalResult(
            rule_id=rule_id,
            declaration_field=decl_field,
            status=status,
            severity=rule.get("severity", "INFORMATIONAL"),
            legal_reference=rule.get("legal_reference", ""),
            official_source_url=rule.get("official_source_url", ""),
            details=details,
        ))

    return results


def compute_overall_status(rule_results: List[RuleEvalResult]) -> str:
    """
    Derive the overall preliminary assessment status from individual rule results.

    Priority:
    1. Any HIGH severity NOT_FOUND → POTENTIAL_NON_COMPLIANCE
    2. Any UNCERTAIN or MANUAL_REVIEW → MANUAL_REVIEW
    3. All FOUND or NOT_APPLICABLE → PASS
    """
    statuses = {r.status for r in rule_results}
    severities_not_found = {
        r.severity for r in rule_results if r.status == DeclarationStatus.NOT_FOUND
    }

    if DeclarationStatus.NOT_FOUND in statuses and "HIGH" in severities_not_found:
        return "POTENTIAL_NON_COMPLIANCE"

    if (DeclarationStatus.UNCERTAIN in statuses
            or DeclarationStatus.MANUAL_REVIEW in statuses
            or (DeclarationStatus.NOT_FOUND in statuses and "MEDIUM" in severities_not_found)):
        return "MANUAL_REVIEW"

    return "PASS"

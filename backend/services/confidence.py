"""
LEGALMET AI — Prototype Confidence Scoring Service
Computes a multi-signal weighted confidence score from pipeline components.

IMPORTANT:
- This is labeled "Prototype Confidence Score" — it is NOT statistically calibrated.
- It does NOT represent probability of legal violation.
- It is a weighted agreement score across OCR, extraction, image quality, and applicability signals.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import List
from backend.database.models import DeclarationStatus
from backend.config import settings


@dataclass
class ConfidenceResult:
    prototype_confidence_score: float
    quality_score: float
    ocr_score: float
    extraction_score: float
    applicability_score: float
    label: str = "Prototype Confidence Score"
    disclaimer: str = (
        "This score reflects internal signal agreement and does not represent "
        "probability of legal violation. It is not statistically calibrated."
    )


def compute_confidence(
    image_quality_score: float,
    ocr_mean_confidence: float,
    extracted_declarations: list,      # List[ExtractedDeclaration]
    rule_results: list,                # List[RuleEvalResult]
) -> ConfidenceResult:
    """
    Compute the Prototype Confidence Score as a weighted sum of:
    - S_quality: normalized image quality score
    - S_ocr: mean OCR confidence across all tokens
    - S_extraction: fraction of applicable declarations extracted at >= 0.65 confidence
    - S_applicability: fraction of rules conclusively resolved (FOUND or NOT_APPLICABLE)
    """
    s_quality = float(max(0.0, min(1.0, image_quality_score)))
    s_ocr = float(max(0.0, min(1.0, ocr_mean_confidence)))

    # Extraction score: fraction of extractors that returned a meaningful result
    if extracted_declarations:
        high_conf = sum(
            1 for d in extracted_declarations
            if d.extracted_value and d.extraction_confidence >= 0.65
        )
        s_extraction = high_conf / len(extracted_declarations)
    else:
        s_extraction = 0.0

    # Applicability score: fraction of rules with conclusive outcome
    conclusive_statuses = {DeclarationStatus.FOUND, DeclarationStatus.NOT_APPLICABLE}
    if rule_results:
        conclusive = sum(1 for r in rule_results if r.status in conclusive_statuses)
        s_applicability = conclusive / len(rule_results)
    else:
        s_applicability = 0.0

    score = (
        settings.conf_weight_quality * s_quality
        + settings.conf_weight_ocr * s_ocr
        + settings.conf_weight_extraction * s_extraction
        + settings.conf_weight_applicability * s_applicability
    )
    score = round(min(1.0, max(0.0, score)), 4)

    return ConfidenceResult(
        prototype_confidence_score=score,
        quality_score=round(s_quality, 4),
        ocr_score=round(s_ocr, 4),
        extraction_score=round(s_extraction, 4),
        applicability_score=round(s_applicability, 4),
    )

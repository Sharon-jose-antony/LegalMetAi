"""
LEGALMET AI — Inspection Pipeline Orchestrator
Chains all services in sequence:
  Image Quality → Preprocessing → Multi-Pass OCR → Declaration Extraction
  → Rule Engine → Evidence → Confidence → Assessment

This is the single entry point called by the /analyze API endpoint.
"""
from __future__ import annotations
import os
import time
import uuid
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import Optional, List

from backend.services.image_quality import assess_quality, ImageQualityResult
from backend.services.preprocessing import preprocess
from backend.services.ocr import (
    OCRProvider, create_ocr_provider, run_multi_pass_ocr, MultiPassOCRResult
)
from backend.services.declaration_extractor import extract_all_declarations, ExtractedDeclaration
from backend.services.rule_engine import evaluate_rules, compute_overall_status, RuleEvalResult
from backend.services.confidence import compute_confidence, ConfidenceResult
from backend.services.evidence import generate_evidence, EvidenceSlice
from backend.database.models import ProductCategory, InspectionStatus, DeclarationStatus
from backend.config import settings


@dataclass
class PipelineResult:
    inspection_id: str
    status: InspectionStatus
    overall_assessment: str            # PASS | MANUAL_REVIEW | POTENTIAL_NON_COMPLIANCE
    quality_result: Optional[ImageQualityResult]
    ocr_result: Optional[MultiPassOCRResult]
    declarations: List[ExtractedDeclaration]
    rule_results: List[RuleEvalResult]
    evidence: List[EvidenceSlice]
    confidence: Optional[ConfidenceResult]
    processing_time_seconds: float
    pipeline_stages: List[str]
    error: Optional[str] = None


# Singleton OCR provider (initialized once per process)
_ocr_provider: Optional[OCRProvider] = None


def get_ocr_provider() -> OCRProvider:
    global _ocr_provider
    if _ocr_provider is None:
        _ocr_provider = create_ocr_provider()
    return _ocr_provider


def run_inspection_pipeline(
    image_path: str,
    product_category: ProductCategory,
    inspection_id: str,
    is_imported: bool = False,
    evidence_save_dir: Optional[str] = None,
) -> PipelineResult:
    """
    Execute the full AI compliance inspection pipeline on a single image.

    Returns a PipelineResult with all intermediate and final outputs.
    No results are fabricated — all data comes from actual image processing.
    """
    t_start = time.perf_counter()
    stages: list[str] = []
    error: Optional[str] = None

    # ── 1. Load Image ─────────────────────────────────────────────────────────
    try:
        pil_image = Image.open(image_path).convert("RGB")
        img_np = np.array(pil_image)
        stages.append("IMAGE_LOADED")
    except Exception as e:
        return PipelineResult(
            inspection_id=inspection_id,
            status=InspectionStatus.ERROR,
            overall_assessment="ERROR",
            quality_result=None,
            ocr_result=None,
            declarations=[],
            rule_results=[],
            evidence=[],
            confidence=None,
            processing_time_seconds=time.perf_counter() - t_start,
            pipeline_stages=["IMAGE_LOAD_FAILED"],
            error=f"Failed to load image: {str(e)}",
        )

    # ── 2. Image Quality Assessment ───────────────────────────────────────────
    quality_result = assess_quality(img_np)
    stages.append("IMAGE_QUALITY_ASSESSED")

    # If image is UNUSABLE, skip OCR entirely
    if quality_result.quality_recommendation == "UNUSABLE":
        stages.append("OCR_SKIPPED_UNUSABLE_IMAGE")
        return PipelineResult(
            inspection_id=inspection_id,
            status=InspectionStatus.MANUAL_REVIEW,
            overall_assessment="MANUAL_REVIEW",
            quality_result=quality_result,
            ocr_result=None,
            declarations=[],
            rule_results=[],
            evidence=[],
            confidence=None,
            processing_time_seconds=round(time.perf_counter() - t_start, 3),
            pipeline_stages=stages,
            error="Image quality is insufficient for automated analysis. Manual inspection required.",
        )

    # ── 3. Preprocessing & Enhancement ────────────────────────────────────────
    preprocessed = preprocess(img_np)
    stages.append("IMAGE_PREPROCESSED")

    # ── 4. Multi-Pass OCR ─────────────────────────────────────────────────────
    try:
        provider = get_ocr_provider()
        ocr_result = run_multi_pass_ocr(
            provider=provider,
            preprocess_result=preprocessed,
        )
        stages.append("OCR_COMPLETE")
    except Exception as e:
        ocr_result = None
        stages.append("OCR_FAILED")
        error = f"OCR failed: {str(e)}"

    # ── 5. Declaration Extraction ──────────────────────────────────────────────
    declarations = []
    if ocr_result:
        full_text = ocr_result.full_text
        tokens = ocr_result.tokens
        declarations = extract_all_declarations(tokens, full_text)
        stages.append("DECLARATIONS_EXTRACTED")
    else:
        stages.append("DECLARATIONS_SKIPPED_OCR_FAILURE")

    # ── 6. Rule Engine Evaluation ──────────────────────────────────────────────
    rule_results = evaluate_rules(
        extracted_declarations=declarations,
        category=product_category,
        quality_score=quality_result.quality_score,
        is_imported=is_imported,
    )
    stages.append("RULES_EVALUATED")

    # ── 7. Evidence Generation ─────────────────────────────────────────────────
    evidence_list: list[EvidenceSlice] = []
    if ocr_result:
        for decl in declarations:
            if decl.extracted_value and decl.raw_ocr_text:
                # Find the rule_id for this field
                matching_rule = next(
                    (r for r in rule_results if r.declaration_field == decl.field), None
                )
                ev = generate_evidence(
                    tokens=ocr_result.tokens,
                    original_image=preprocessed.original,
                    declaration_field=decl.field,
                    matched_text=decl.raw_ocr_text,
                    rule_id=matching_rule.rule_id if matching_rule else None,
                    inspection_id=inspection_id,
                    save_dir=evidence_save_dir,
                )
                if ev:
                    evidence_list.append(ev)
    stages.append("EVIDENCE_GENERATED")

    # ── 8. Confidence Scoring ─────────────────────────────────────────────────
    confidence = compute_confidence(
        image_quality_score=quality_result.quality_score,
        ocr_mean_confidence=ocr_result.mean_confidence if ocr_result else 0.0,
        extracted_declarations=declarations,
        rule_results=rule_results,
    )
    stages.append("CONFIDENCE_COMPUTED")

    # ── 9. Overall Assessment ─────────────────────────────────────────────────
    overall = compute_overall_status(rule_results)
    status_map = {
        "PASS": InspectionStatus.PASS,
        "MANUAL_REVIEW": InspectionStatus.MANUAL_REVIEW,
        "POTENTIAL_NON_COMPLIANCE": InspectionStatus.POTENTIAL_NON_COMPLIANCE,
    }
    final_status = status_map.get(overall, InspectionStatus.MANUAL_REVIEW)
    stages.append("ASSESSMENT_COMPLETE")

    return PipelineResult(
        inspection_id=inspection_id,
        status=final_status,
        overall_assessment=overall,
        quality_result=quality_result,
        ocr_result=ocr_result,
        declarations=declarations,
        rule_results=rule_results,
        evidence=evidence_list,
        confidence=confidence,
        processing_time_seconds=round(time.perf_counter() - t_start, 3),
        pipeline_stages=stages,
        error=error,
    )

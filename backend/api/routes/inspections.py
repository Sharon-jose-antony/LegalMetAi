"""
LEGALMET AI — Inspections API Routes
POST   /api/inspections                     — Create new inspection
POST   /api/inspections/{id}/upload         — Upload package image
POST   /api/inspections/{id}/quality-check  — Run quality assessment
POST   /api/inspections/{id}/analyze        — Run full AI pipeline
GET    /api/inspections/{id}                — Get inspection detail
GET    /api/inspections                     — List/search inspections
GET    /api/inspections/{id}/evidence       — Get evidence slices
POST   /api/inspections/{id}/review         — Submit manual review
GET    /api/inspections/{id}/report.pdf     — Download PDF report
"""
import os
import shutil
import uuid
from datetime import datetime
from typing import Optional, List

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status, Query
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from backend.database.database import get_db
from backend.database.models import (
    User, Inspection, InspectionImage, OCRResult as OCRResultModel,
    Declaration, RuleResult, Evidence as EvidenceModel,
    ManualReview, Report, ProductCategory, InspectionStatus,
    DeclarationStatus, ReviewDecision
)
from backend.schemas.schemas import (
    InspectionCreate, InspectionSummary, InspectionDetail,
    AnalysisResponse, ManualReviewCreate, ManualReviewRead,
    ImageQualityReport, EvidenceSchema, ConfidenceBreakdown,
    DeclarationResult, RuleResultSchema
)
from backend.auth.dependencies import get_current_user
from backend.services.image_quality import assess_quality
from backend.services.pipeline import run_inspection_pipeline
from backend.reports.report_generator import generate_pdf_report
from backend.config import settings
from PIL import Image

router = APIRouter(prefix="/api/inspections", tags=["Inspections"])


def _generate_inspection_id(db: Session) -> str:
    """Generate LM-YYYY-XXXXXX format inspection ID."""
    year = datetime.utcnow().year
    count = db.query(Inspection).count() + 1
    return f"LM-{year}-{count:06d}"


def _validate_file(file: UploadFile):
    ext = file.filename.rsplit(".", 1)[-1].lower() if "." in file.filename else ""
    if ext not in settings.allowed_extensions_list:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported file type '.{ext}'. Allowed: {settings.allowed_extensions}",
        )


# ── Create Inspection ─────────────────────────────────────────────────────────

@router.post("", response_model=InspectionSummary, status_code=status.HTTP_201_CREATED)
def create_inspection(
    data: InspectionCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    inspection_id = _generate_inspection_id(db)
    inspection = Inspection(
        inspection_id=inspection_id,
        inspector_id=current_user.id,
        product_name=data.product_name,
        brand_name=data.brand_name,
        product_category=data.product_category,
        status=InspectionStatus.DRAFT,
        is_demo=data.is_demo,
        demo_label=data.demo_label,
        notes=data.notes,
    )
    db.add(inspection)
    db.commit()
    db.refresh(inspection)
    result = InspectionSummary.model_validate(inspection)
    result.inspector_name = current_user.full_name
    return result


# ── Upload Image ──────────────────────────────────────────────────────────────

@router.post("/{inspection_id}/upload", response_model=ImageQualityReport)
def upload_image(
    inspection_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    inspection = db.query(Inspection).filter(Inspection.inspection_id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")

    _validate_file(file)

    # Save uploaded file
    safe_name = f"{inspection_id}_original_{uuid.uuid4().hex[:8]}.{file.filename.rsplit('.', 1)[-1].lower()}"
    save_path = os.path.join(settings.upload_dir, safe_name)
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Quick quality check on upload
    pil_img = Image.open(save_path).convert("RGB")
    quality = assess_quality(pil_img)

    img_record = InspectionImage(
        inspection_id=inspection.id,
        image_type="original",
        file_path=save_path,
        file_name=safe_name,
        width_px=quality.width_px,
        height_px=quality.height_px,
        blur_score=quality.blur_score,
        contrast_score=quality.contrast_score,
        glare_fraction=quality.glare_fraction,
        quality_score=quality.quality_score,
        quality_recommendation=quality.quality_recommendation,
    )
    db.add(img_record)
    db.commit()

    return ImageQualityReport(
        quality_score=quality.quality_score,
        resolution_ok=quality.resolution_ok,
        width_px=quality.width_px,
        height_px=quality.height_px,
        blur_score=quality.blur_score,
        blur_detected=quality.blur_detected,
        contrast_score=quality.contrast_score,
        contrast_ok=quality.contrast_ok,
        glare_fraction=quality.glare_fraction,
        glare_detected=quality.glare_detected,
        quality_recommendation=quality.quality_recommendation,
        message=quality.message,
    )


# ── Run Full Analysis ─────────────────────────────────────────────────────────

@router.post("/{inspection_id}/analyze", response_model=AnalysisResponse)
def analyze_inspection(
    inspection_id: str,
    is_imported: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    inspection = db.query(Inspection).filter(Inspection.inspection_id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")

    # Find the original uploaded image
    img_record = db.query(InspectionImage).filter(
        InspectionImage.inspection_id == inspection.id,
        InspectionImage.image_type == "original"
    ).order_by(InspectionImage.id.desc()).first()

    if not img_record:
        raise HTTPException(status_code=400, detail="No image uploaded for this inspection. Upload an image first.")

    # Update status to PROCESSING
    inspection.status = InspectionStatus.PROCESSING
    db.commit()

    # Evidence save directory
    evidence_dir = os.path.join(settings.upload_dir, "evidence", inspection_id)

    # Run full AI pipeline
    result = run_inspection_pipeline(
        image_path=img_record.file_path,
        product_category=inspection.product_category,
        inspection_id=inspection_id,
        is_imported=is_imported,
        evidence_save_dir=evidence_dir,
    )

    # Persist results to database
    inspection.status = result.status
    inspection.overall_confidence = (
        result.confidence.prototype_confidence_score if result.confidence else None
    )
    inspection.completed_at = datetime.utcnow()

    # Clean up any existing results from previous run
    db.query(OCRResultModel).filter(OCRResultModel.inspection_id == inspection.id).delete()
    db.query(Declaration).filter(Declaration.inspection_id == inspection.id).delete()
    db.query(RuleResult).filter(RuleResult.inspection_id == inspection.id).delete()
    db.query(EvidenceModel).filter(EvidenceModel.inspection_id == inspection.id).delete()
    db.flush()

    # Save OCR results
    if result.ocr_result:
        for token in result.ocr_result.tokens:
            ocr_row = OCRResultModel(
                inspection_id=inspection.id,
                source_image_id=img_record.id,
                ocr_pass=token.ocr_pass,
                text=token.text,
                confidence=token.confidence,
                bbox_ymin=token.bbox[0] if token.bbox else None,
                bbox_xmin=token.bbox[1] if token.bbox else None,
                bbox_ymax=token.bbox[2] if token.bbox else None,
                bbox_xmax=token.bbox[3] if token.bbox else None,
                polygon_json=token.polygon,
            )
            db.add(ocr_row)

    # Save declarations
    for decl in result.declarations:
        decl_status = _map_decl_status(decl)
        d_row = Declaration(
            inspection_id=inspection.id,
            field=decl.field,
            status=decl_status,
            extracted_value=decl.extracted_value,
            raw_ocr_text=decl.raw_ocr_text,
            normalized_value=decl.normalized_value,
            extraction_confidence=decl.extraction_confidence,
        )
        db.add(d_row)

    # Save rule results
    for rr in result.rule_results:
        rr_row = RuleResult(
            inspection_id=inspection.id,
            rule_id=rr.rule_id,
            declaration_field=rr.declaration_field,
            status=rr.status,
            severity=rr.severity,
            legal_reference=rr.legal_reference,
            official_source_url=rr.official_source_url,
            details=rr.details,
        )
        db.add(rr_row)

    # Save evidence
    for ev in result.evidence:
        ev_row = EvidenceModel(
            inspection_id=inspection.id,
            source_image_id=img_record.id,
            declaration_field=ev.declaration_field,
            crop_file_path=ev.crop_file_path,
            bbox_json=ev.bbox,
            ocr_text=ev.ocr_text,
            confidence=ev.confidence,
            rule_id=ev.rule_id,
        )
        db.add(ev_row)

    db.commit()

    # Build response
    conf_breakdown = None
    if result.confidence:
        conf_breakdown = ConfidenceBreakdown(
            prototype_confidence_score=result.confidence.prototype_confidence_score,
            quality_score=result.confidence.quality_score,
            ocr_score=result.confidence.ocr_score,
            extraction_score=result.confidence.extraction_score,
            applicability_score=result.confidence.applicability_score,
        )

    return AnalysisResponse(
        inspection_id=inspection_id,
        status=result.status,
        overall_assessment=result.overall_assessment,
        confidence_breakdown=conf_breakdown,
        declarations=[
            DeclarationResult(
                field=d.field,
                status=_map_decl_status(d),
                extracted_value=d.extracted_value,
                raw_ocr_text=d.raw_ocr_text,
                normalized_value=d.normalized_value,
                extraction_confidence=d.extraction_confidence,
            ) for d in result.declarations
        ],
        rule_results=[
            RuleResultSchema(
                rule_id=r.rule_id,
                declaration_field=r.declaration_field,
                status=r.status,
                severity=r.severity,
                legal_reference=r.legal_reference,
                official_source_url=r.official_source_url,
                details=r.details,
            ) for r in result.rule_results
        ],
        processing_time_seconds=result.processing_time_seconds,
        pipeline_stages=result.pipeline_stages,
    )


def _map_decl_status(decl) -> DeclarationStatus:
    if decl.extracted_value and decl.extraction_confidence >= 0.65:
        return DeclarationStatus.FOUND
    elif decl.extracted_value and decl.extraction_confidence >= 0.30:
        return DeclarationStatus.UNCERTAIN
    else:
        return DeclarationStatus.NOT_FOUND


# ── Get Inspection Detail ─────────────────────────────────────────────────────

@router.get("/{inspection_id}", response_model=InspectionDetail)
def get_inspection(
    inspection_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    inspection = db.query(Inspection).filter(Inspection.inspection_id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")

    declarations = [
        DeclarationResult(
            field=d.field,
            status=d.status,
            extracted_value=d.extracted_value,
            raw_ocr_text=d.raw_ocr_text,
            normalized_value=d.normalized_value,
            extraction_confidence=d.extraction_confidence,
        )
        for d in inspection.declarations
    ]

    rule_results = [
        RuleResultSchema(
            rule_id=r.rule_id,
            declaration_field=r.declaration_field,
            status=r.status,
            severity=r.severity,
            legal_reference=r.legal_reference,
            official_source_url=r.official_source_url,
            details=r.details,
        )
        for r in inspection.rule_results
    ]

    return InspectionDetail(
        id=inspection.id,
        inspection_id=inspection.inspection_id,
        product_name=inspection.product_name,
        brand_name=inspection.brand_name,
        product_category=inspection.product_category,
        status=inspection.status,
        overall_confidence=inspection.overall_confidence,
        is_demo=inspection.is_demo,
        demo_label=inspection.demo_label,
        notes=inspection.notes,
        created_at=inspection.created_at,
        completed_at=inspection.completed_at,
        declarations=declarations,
        rule_results=rule_results,
    )


# ── List Inspections ──────────────────────────────────────────────────────────

@router.get("", response_model=List[InspectionSummary])
def list_inspections(
    status_filter: Optional[str] = Query(None, alias="status"),
    category: Optional[str] = None,
    search: Optional[str] = None,
    limit: int = Query(50, le=200),
    offset: int = 0,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    q = db.query(Inspection)
    if status_filter:
        try:
            q = q.filter(Inspection.status == InspectionStatus(status_filter))
        except ValueError:
            pass
    if category:
        try:
            q = q.filter(Inspection.product_category == ProductCategory(category))
        except ValueError:
            pass
    if search:
        pattern = f"%{search}%"
        q = q.filter(
            (Inspection.inspection_id.ilike(pattern)) |
            (Inspection.product_name.ilike(pattern)) |
            (Inspection.brand_name.ilike(pattern))
        )
    inspections = q.order_by(Inspection.created_at.desc()).offset(offset).limit(limit).all()
    results = []
    for i in inspections:
        s = InspectionSummary.model_validate(i)
        s.inspector_name = i.inspector.full_name if i.inspector else None
        results.append(s)
    return results


# ── Get Evidence ──────────────────────────────────────────────────────────────

@router.get("/{inspection_id}/evidence", response_model=List[EvidenceSchema])
def get_evidence(
    inspection_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    inspection = db.query(Inspection).filter(Inspection.inspection_id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")
    evidence = db.query(EvidenceModel).filter(EvidenceModel.inspection_id == inspection.id).all()
    return [
        EvidenceSchema(
            id=e.id,
            declaration_field=e.declaration_field,
            crop_file_path=e.crop_file_path,
            bbox=e.bbox_json,
            ocr_text=e.ocr_text,
            confidence=e.confidence,
            rule_id=e.rule_id,
        )
        for e in evidence
    ]


# ── Get OCR Raw Tokens ─────────────────────────────────────────────────────────

@router.get("/{inspection_id}/ocr")
def get_inspection_ocr(
    inspection_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    inspection = db.query(Inspection).filter(Inspection.inspection_id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")
    tokens = db.query(OCRResultModel).filter(OCRResultModel.inspection_id == inspection.id).order_by(OCRResultModel.confidence.desc()).all()
    return [
        {
            "id": t.id,
            "text": t.text,
            "confidence": t.confidence,
            "ocr_pass": t.ocr_pass,
            "bbox": [t.bbox_ymin, t.bbox_xmin, t.bbox_ymax, t.bbox_xmax],
        }
        for t in tokens
    ]


# ── Submit Manual Review ──────────────────────────────────────────────────────

@router.post("/{inspection_id}/review", response_model=ManualReviewRead, status_code=201)
def submit_review(
    inspection_id: str,
    review: ManualReviewCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    inspection = db.query(Inspection).filter(Inspection.inspection_id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")

    existing = db.query(ManualReview).filter(ManualReview.inspection_id == inspection.id).first()
    if existing:
        existing.decision = review.decision
        existing.statutory_notes = review.statutory_notes
        existing.flagged_declarations = review.flagged_declarations
        existing.reviewed_at = datetime.utcnow()
        existing.reviewer_id = current_user.id
        db.commit()
        db.refresh(existing)
        return ManualReviewRead.model_validate(existing)

    review_row = ManualReview(
        inspection_id=inspection.id,
        reviewer_id=current_user.id,
        decision=review.decision,
        statutory_notes=review.statutory_notes,
        flagged_declarations=review.flagged_declarations,
        reviewed_at=datetime.utcnow(),
    )
    db.add(review_row)
    db.commit()
    db.refresh(review_row)
    return ManualReviewRead.model_validate(review_row)


# ── Download PDF Report ────────────────────────────────────────────────────────

@router.get("/{inspection_id}/report.pdf")
def download_report(
    inspection_id: str,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    inspection = db.query(Inspection).filter(Inspection.inspection_id == inspection_id).first()
    if not inspection:
        raise HTTPException(status_code=404, detail="Inspection not found")

    # Generate or return cached report
    report_row = db.query(Report).filter(Report.inspection_id == inspection.id).first()

    if not report_row or not os.path.exists(report_row.file_path):
        pdf_path = generate_pdf_report(inspection, db, settings.reports_dir)
        if not report_row:
            report_row = Report(
                inspection_id=inspection.id,
                file_path=pdf_path,
                file_name=os.path.basename(pdf_path),
            )
            db.add(report_row)
        else:
            report_row.file_path = pdf_path
            report_row.file_name = os.path.basename(pdf_path)
            report_row.generated_at = datetime.utcnow()
        db.commit()

    return FileResponse(
        path=report_row.file_path,
        media_type="application/pdf",
        filename=report_row.file_name,
    )

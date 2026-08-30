"""
LEGALMET AI — Pydantic Schemas
Request/Response models for all API endpoints.
"""
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Any
from datetime import datetime
from backend.database.models import (
    UserRole, ProductCategory, InspectionStatus,
    DeclarationStatus, ReviewDecision
)


# ── Auth ──────────────────────────────────────────────────────────────────────

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: int
    username: str
    role: UserRole


class LoginRequest(BaseModel):
    username: str
    password: str


class UserCreate(BaseModel):
    username: str = Field(..., min_length=3, max_length=64)
    full_name: str = Field(..., min_length=2, max_length=128)
    badge_number: Optional[str] = None
    department: Optional[str] = None
    role: UserRole = UserRole.INSPECTOR
    password: str = Field(..., min_length=8)


class UserRead(BaseModel):
    id: int
    username: str
    full_name: str
    badge_number: Optional[str]
    department: Optional[str]
    role: UserRole
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Image Quality ─────────────────────────────────────────────────────────────

class ImageQualityReport(BaseModel):
    quality_score: float = Field(..., ge=0.0, le=1.0)
    resolution_ok: bool
    width_px: int
    height_px: int
    blur_score: float
    blur_detected: bool
    contrast_score: float
    contrast_ok: bool
    glare_fraction: float
    glare_detected: bool
    quality_recommendation: str  # GOOD, ACCEPTABLE, POOR, UNUSABLE
    message: str


# ── OCR ───────────────────────────────────────────────────────────────────────

class OCRResultSchema(BaseModel):
    text: str
    confidence: float
    bbox: Optional[List[int]] = None   # [ymin, xmin, ymax, xmax]
    polygon: Optional[List[List[int]]] = None
    ocr_pass: str                       # original, enhanced_clahe, high_contrast


# ── Declarations ──────────────────────────────────────────────────────────────

class DeclarationResult(BaseModel):
    field: str
    status: DeclarationStatus
    extracted_value: Optional[str]
    raw_ocr_text: Optional[str]
    normalized_value: Optional[str]
    extraction_confidence: Optional[float]


# ── Rule Results ──────────────────────────────────────────────────────────────

class RuleResultSchema(BaseModel):
    rule_id: str
    declaration_field: str
    status: DeclarationStatus
    severity: Optional[str]
    legal_reference: Optional[str]
    official_source_url: Optional[str]
    details: Optional[str]


# ── Evidence ──────────────────────────────────────────────────────────────────

class EvidenceSchema(BaseModel):
    id: int
    declaration_field: str
    crop_file_path: Optional[str]
    bbox: Optional[List[int]]
    ocr_text: Optional[str]
    confidence: Optional[float]
    rule_id: Optional[str]

    model_config = {"from_attributes": True}


# ── Confidence ────────────────────────────────────────────────────────────────

class ConfidenceBreakdown(BaseModel):
    prototype_confidence_score: float = Field(..., ge=0.0, le=1.0)
    quality_score: float
    ocr_score: float
    extraction_score: float
    applicability_score: float
    label: str = "Prototype Confidence Score"
    disclaimer: str = (
        "This score reflects internal signal agreement and does not represent "
        "probability of legal violation. It is not statistically calibrated."
    )


from pydantic import BaseModel, Field, EmailStr, field_validator


class InspectionCreate(BaseModel):
    product_name: Optional[str] = None
    brand_name: Optional[str] = None
    product_category: ProductCategory
    notes: Optional[str] = None
    is_demo: bool = False
    demo_label: Optional[str] = None

    @field_validator("product_category", mode="before")
    @classmethod
    def normalize_category(cls, v):
        if isinstance(v, str):
            clean = v.strip().upper().replace(" ", "_").replace("-", "_")
            if clean in ProductCategory.__members__:
                return ProductCategory[clean]
            # Map common variations
            for member in ProductCategory:
                if member.value == clean or member.name == clean:
                    return member
        return v


class InspectionSummary(BaseModel):
    id: int
    inspection_id: str
    product_name: Optional[str]
    brand_name: Optional[str]
    product_category: ProductCategory
    status: InspectionStatus
    overall_confidence: Optional[float]
    is_demo: bool
    demo_label: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    inspector_name: Optional[str] = None

    model_config = {"from_attributes": True}


class InspectionDetail(BaseModel):
    id: int
    inspection_id: str
    product_name: Optional[str]
    brand_name: Optional[str]
    product_category: ProductCategory
    status: InspectionStatus
    overall_confidence: Optional[float]
    is_demo: bool
    demo_label: Optional[str]
    notes: Optional[str]
    created_at: datetime
    completed_at: Optional[datetime]
    declarations: List[DeclarationResult] = []
    rule_results: List[RuleResultSchema] = []
    confidence_breakdown: Optional[ConfidenceBreakdown] = None

    model_config = {"from_attributes": True}


class AnalysisResponse(BaseModel):
    inspection_id: str
    status: InspectionStatus
    overall_assessment: str
    confidence_breakdown: ConfidenceBreakdown
    declarations: List[DeclarationResult]
    rule_results: List[RuleResultSchema]
    processing_time_seconds: float
    pipeline_stages: List[str]
    disclaimer: str = (
        "AI-assisted preliminary assessment based on submitted image evidence. "
        "This system does not replace statutory inspection, measurement or legal "
        "determination by an authorized officer."
    )


# ── Manual Review ─────────────────────────────────────────────────────────────

class ManualReviewCreate(BaseModel):
    decision: ReviewDecision
    statutory_notes: Optional[str] = None
    flagged_declarations: Optional[List[str]] = None


class ManualReviewRead(BaseModel):
    id: int
    decision: ReviewDecision
    statutory_notes: Optional[str]
    flagged_declarations: Optional[List[str]]
    reviewed_at: Optional[datetime]

    model_config = {"from_attributes": True}


# ── Dashboard ─────────────────────────────────────────────────────────────────

class DashboardStats(BaseModel):
    total_inspections: int
    pass_count: int
    potential_non_compliance_count: int
    manual_review_count: int
    draft_count: int
    average_confidence: Optional[float]
    inspections_by_category: dict
    recent_inspections: List[InspectionSummary] = []
    disclaimer: str = (
        "Statistics are derived from inspections performed by this system. "
        "They represent AI-assisted preliminary assessments only."
    )

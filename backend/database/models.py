"""
LEGALMET AI — SQLAlchemy ORM Models (SQLite)
Defines the relational database schema for the inspection platform.
"""
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime,
    Text, ForeignKey, JSON, Enum as SAEnum
)
from sqlalchemy.orm import relationship, declarative_base
import enum

Base = declarative_base()


# ── Enumerations ──────────────────────────────────────────────────────────────

class UserRole(str, enum.Enum):
    ADMIN = "ADMIN"
    INSPECTOR = "INSPECTOR"
    VIEWER = "VIEWER"


class ProductCategory(str, enum.Enum):
    PACKAGED_FOOD = "PACKAGED_FOOD"
    COSMETICS = "COSMETICS"
    HOUSEHOLD_COMMODITY = "HOUSEHOLD_COMMODITY"


class InspectionStatus(str, enum.Enum):
    DRAFT = "DRAFT"
    PROCESSING = "PROCESSING"
    PASS = "PASS"
    POTENTIAL_NON_COMPLIANCE = "POTENTIAL_NON_COMPLIANCE"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    ERROR = "ERROR"


class DeclarationStatus(str, enum.Enum):
    FOUND = "FOUND"
    NOT_FOUND = "NOT_FOUND"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNCERTAIN = "UNCERTAIN"
    MANUAL_REVIEW = "MANUAL_REVIEW"


class ReviewDecision(str, enum.Enum):
    CONFIRMED = "CONFIRMED"
    OVERRIDDEN_PASS = "OVERRIDDEN_PASS"
    OVERRIDDEN_NON_COMPLIANCE = "OVERRIDDEN_NON_COMPLIANCE"
    NEEDS_FURTHER_INSPECTION = "NEEDS_FURTHER_INSPECTION"


# ── Tables ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    full_name = Column(String(128), nullable=False)
    badge_number = Column(String(32), unique=True, nullable=True)
    department = Column(String(128), nullable=True)
    role = Column(SAEnum(UserRole), nullable=False, default=UserRole.INSPECTOR)
    hashed_password = Column(String(256), nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    inspections = relationship("Inspection", back_populates="inspector")
    reviews = relationship("ManualReview", back_populates="reviewer")


class Inspection(Base):
    __tablename__ = "inspections"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(String(32), unique=True, nullable=False, index=True)  # LM-2026-000001
    inspector_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    product_name = Column(String(256), nullable=True)
    brand_name = Column(String(128), nullable=True)
    product_category = Column(SAEnum(ProductCategory), nullable=False)
    status = Column(SAEnum(InspectionStatus), nullable=False, default=InspectionStatus.DRAFT)
    overall_confidence = Column(Float, nullable=True)
    notes = Column(Text, nullable=True)
    is_demo = Column(Boolean, default=False)   # True for controlled demo test cases
    demo_label = Column(String(128), nullable=True)  # e.g. "CONTROLLED DEMO TEST CASE"
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    inspector = relationship("User", back_populates="inspections")
    images = relationship("InspectionImage", back_populates="inspection", cascade="all, delete-orphan")
    ocr_results = relationship("OCRResult", back_populates="inspection", cascade="all, delete-orphan")
    declarations = relationship("Declaration", back_populates="inspection", cascade="all, delete-orphan")
    rule_results = relationship("RuleResult", back_populates="inspection", cascade="all, delete-orphan")
    evidence = relationship("Evidence", back_populates="inspection", cascade="all, delete-orphan")
    manual_review = relationship("ManualReview", back_populates="inspection", uselist=False, cascade="all, delete-orphan")
    report = relationship("Report", back_populates="inspection", uselist=False, cascade="all, delete-orphan")


class InspectionImage(Base):
    __tablename__ = "inspection_images"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id"), nullable=False)
    image_type = Column(String(32), nullable=False)  # original, enhanced_clahe, high_contrast, crop_*
    file_path = Column(String(512), nullable=False)
    file_name = Column(String(256), nullable=False)
    width_px = Column(Integer, nullable=True)
    height_px = Column(Integer, nullable=True)
    # Image quality metrics
    blur_score = Column(Float, nullable=True)
    contrast_score = Column(Float, nullable=True)
    glare_fraction = Column(Float, nullable=True)
    quality_score = Column(Float, nullable=True)
    quality_recommendation = Column(String(32), nullable=True)  # GOOD, ACCEPTABLE, POOR, UNUSABLE
    created_at = Column(DateTime, default=datetime.utcnow)

    inspection = relationship("Inspection", back_populates="images")
    ocr_results = relationship("OCRResult", back_populates="source_image")
    evidence = relationship("Evidence", back_populates="source_image")


class OCRResult(Base):
    __tablename__ = "ocr_results"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id"), nullable=False)
    source_image_id = Column(Integer, ForeignKey("inspection_images.id"), nullable=False)
    ocr_pass = Column(String(32), nullable=False)       # original, enhanced_clahe, high_contrast
    text = Column(Text, nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_ymin = Column(Integer, nullable=True)
    bbox_xmin = Column(Integer, nullable=True)
    bbox_ymax = Column(Integer, nullable=True)
    bbox_xmax = Column(Integer, nullable=True)
    polygon_json = Column(JSON, nullable=True)           # Full polygon points
    created_at = Column(DateTime, default=datetime.utcnow)

    inspection = relationship("Inspection", back_populates="ocr_results")
    source_image = relationship("InspectionImage", back_populates="ocr_results")


class Declaration(Base):
    __tablename__ = "declarations"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id"), nullable=False)
    field = Column(String(64), nullable=False)           # e.g. MRP, NET_QUANTITY
    status = Column(SAEnum(DeclarationStatus), nullable=False)
    extracted_value = Column(String(512), nullable=True)
    raw_ocr_text = Column(Text, nullable=True)           # Exact OCR snippet preserved
    normalized_value = Column(String(512), nullable=True)
    extraction_confidence = Column(Float, nullable=True)
    ocr_result_id = Column(Integer, ForeignKey("ocr_results.id"), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    inspection = relationship("Inspection", back_populates="declarations")


class RuleResult(Base):
    __tablename__ = "rule_results"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id"), nullable=False)
    rule_id = Column(String(64), nullable=False)
    declaration_field = Column(String(64), nullable=False)
    status = Column(SAEnum(DeclarationStatus), nullable=False)
    severity = Column(String(16), nullable=True)         # HIGH, MEDIUM, INFORMATIONAL
    legal_reference = Column(String(256), nullable=True)
    official_source_url = Column(String(512), nullable=True)
    details = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    inspection = relationship("Inspection", back_populates="rule_results")


class Evidence(Base):
    __tablename__ = "evidence"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id"), nullable=False)
    source_image_id = Column(Integer, ForeignKey("inspection_images.id"), nullable=False)
    declaration_field = Column(String(64), nullable=False)
    crop_file_path = Column(String(512), nullable=True)  # Cropped image slice with bbox highlight
    bbox_json = Column(JSON, nullable=True)
    ocr_text = Column(Text, nullable=True)
    confidence = Column(Float, nullable=True)
    rule_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    inspection = relationship("Inspection", back_populates="evidence")
    source_image = relationship("InspectionImage", back_populates="evidence")


class ManualReview(Base):
    __tablename__ = "manual_reviews"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id"), nullable=False, unique=True)
    reviewer_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    decision = Column(SAEnum(ReviewDecision), nullable=True)
    statutory_notes = Column(Text, nullable=True)
    flagged_declarations = Column(JSON, nullable=True)  # List of field names inspector flagged
    reviewed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    inspection = relationship("Inspection", back_populates="manual_review")
    reviewer = relationship("User", back_populates="reviews")


class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    inspection_id = Column(Integer, ForeignKey("inspections.id"), nullable=False, unique=True)
    file_path = Column(String(512), nullable=False)
    file_name = Column(String(256), nullable=False)
    generated_at = Column(DateTime, default=datetime.utcnow)

    inspection = relationship("Inspection", back_populates="report")

"""
LEGALMET AI — Dashboard Analytics API
GET /api/dashboard/stats
GET /api/rules
"""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from backend.database.database import get_db
from backend.database.models import Inspection, InspectionStatus, ProductCategory
from backend.schemas.schemas import DashboardStats, InspectionSummary
from backend.auth.dependencies import get_current_user
from backend.database.models import User
import json, os

router = APIRouter(prefix="/api", tags=["Dashboard & Rules"])

RULES_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "rules", "legal_metrology_rules.json")


@router.get("/dashboard/stats", response_model=DashboardStats)
def dashboard_stats(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    total = db.query(Inspection).count()
    pass_c = db.query(Inspection).filter(Inspection.status == InspectionStatus.PASS).count()
    nc_c = db.query(Inspection).filter(Inspection.status == InspectionStatus.POTENTIAL_NON_COMPLIANCE).count()
    mr_c = db.query(Inspection).filter(Inspection.status == InspectionStatus.MANUAL_REVIEW).count()
    draft_c = db.query(Inspection).filter(Inspection.status == InspectionStatus.DRAFT).count()

    avg_conf = db.query(func.avg(Inspection.overall_confidence)).filter(
        Inspection.overall_confidence.isnot(None)
    ).scalar()

    by_cat = {}
    for cat in ProductCategory:
        by_cat[cat.value] = db.query(Inspection).filter(Inspection.product_category == cat).count()

    recent = (
        db.query(Inspection)
        .order_by(Inspection.created_at.desc())
        .limit(10)
        .all()
    )
    recent_summaries = []
    for i in recent:
        s = InspectionSummary.model_validate(i)
        s.inspector_name = i.inspector.full_name if i.inspector else None
        recent_summaries.append(s)

    return DashboardStats(
        total_inspections=total,
        pass_count=pass_c,
        potential_non_compliance_count=nc_c,
        manual_review_count=mr_c,
        draft_count=draft_c,
        average_confidence=round(float(avg_conf), 4) if avg_conf else None,
        inspections_by_category=by_cat,
        recent_inspections=recent_summaries,
    )


@router.get("/rules")
def get_rules(current_user: User = Depends(get_current_user)):
    path = os.path.abspath(RULES_PATH)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/health")
def health():
    return {"status": "ok", "service": "LEGALMET AI", "version": "1.0.0-mvp"}

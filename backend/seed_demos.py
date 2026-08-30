"""
LEGALMET AI — Seed Demo Scenarios
Generates synthetic label images and runs the 3 controlled demo scenarios:
  1. DEMO 1: High Compliance Packaged Food (PASS)
  2. DEMO 2: Difficult Low-Contrast / Glare Cosmetics (MANUAL_REVIEW)
  3. DEMO 3: Deliberately Altered Non-Compliant Household Commodity (POTENTIAL_NON_COMPLIANCE)
"""
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.database.database import SessionLocal, init_db
from backend.database.models import User, UserRole, ProductCategory, InspectionStatus
from backend.services.pipeline import run_inspection_pipeline
from backend.services.image_quality import assess_quality
from backend.api.routes.inspections import _map_decl_status
from backend.database.models import (
    Inspection, InspectionImage, OCRResult as OCRResultModel,
    Declaration, RuleResult, Evidence as EvidenceModel
)
from backend.config import settings
from datetime import datetime


def create_demo_image_1_compliant_food(path: str):
    """
    DEMO 1 — High-resolution clear package label with all mandatory declarations.
    """
    img = Image.new("RGB", (1200, 900), color=(248, 246, 240))
    draw = ImageDraw.Draw(img)

    # Decorative header
    draw.rectangle([(40, 40), (1160, 140)], fill=(34, 70, 44))
    draw.text((60, 60), "NUTRI-DELIGHT WHOLE WHEAT BISCUITS", fill=(255, 255, 255))
    draw.text((60, 95), "High Fibre Digestive Biscuits", fill=(200, 230, 200))

    # Declarations Panel
    y = 170
    lines = [
        "Common / Generic Name: Whole Wheat Biscuits",
        "Net Quantity: 500 g (Contains 2 packs of 250 g each)",
        "MRP: Rs. 120.00 (inclusive of all taxes)",
        "Unit Sale Price: Rs. 0.24 / g",
        "Month & Year of Manufacture: AUG 2026",
        "Best Before: 6 Months from Packaging / FEB 2027",
        "Manufactured & Packed by: NutriBake Foods India Pvt. Ltd.",
        "Address: Plot 42, Industrial Area, Phase II, Pune, Maharashtra 411019",
        "Consumer Care Details:",
        "  Helpline / Toll-Free: 1800-222-3344",
        "  Email: customercare@nutribake.in",
        "  Grievance Officer: Plot 42, Industrial Area, Pune 411019",
        "Country of Origin: India",
    ]
    for line in lines:
        draw.text((60, y), line, fill=(20, 25, 35))
        y += 48

    # Border
    draw.rectangle([(30, 30), (1170, 870)], outline=(120, 140, 120), width=3)
    img.save(path, "PNG")
    print(f"Created Demo 1 image: {path}")


def create_demo_image_2_glare_cosmetic(path: str):
    """
    DEMO 2 — Cosmetic product with glare and lower contrast on consumer care text.
    """
    img = Image.new("RGB", (1000, 800), color=(230, 225, 235))
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([(40, 40), (960, 130)], fill=(70, 30, 80))
    draw.text((60, 65), "GLOW-RADIANCE NOURISHING FACE SERUM", fill=(255, 255, 255))

    y = 160
    lines = [
        "Generic Name: Face Serum with Vitamin C & E",
        "Net Quantity: 30 ml",
        "MRP: Rs. 499.00 (inclusive of all taxes)",
        "Mfg Date: 05/2026",
        "Use Before: 24 months from Mfg",
        "Manufactured by: Aura Cosmeceuticals Ltd.",
        "Address: B-12, Sector 63, Noida, UP 201301",
    ]
    for line in lines:
        draw.text((60, y), line, fill=(30, 20, 40))
        y += 45

    # Faint text for Consumer Care (to induce lower OCR confidence / glare flag)
    draw.text((60, y), "Consumer Feedback: care@auracosmetics.in", fill=(180, 175, 185))

    # Convert to numpy for synthetic glare patch
    np_img = np.array(img)
    # Add a simulated reflective glare spot
    cv2.circle(np_img, (750, 400), 120, (255, 255, 255), -1)
    np_img = cv2.GaussianBlur(np_img, (21, 21), 0)
    
    res = Image.fromarray(np_img)
    res.save(path, "PNG")
    print(f"Created Demo 2 image: {path}")


def create_demo_image_3_missing_declarations(path: str):
    """
    DEMO 3 — Deliberately altered non-compliant household detergent:
    Missing Consumer Care helpline/email, and missing Unit Sale Price (USP).
    """
    img = Image.new("RGB", (1100, 800), color=(240, 245, 250))
    draw = ImageDraw.Draw(img)

    # Header
    draw.rectangle([(40, 40), (1060, 130)], fill=(20, 50, 100))
    draw.text((60, 65), "SPARKLE ULTRA DISHWASH GEL", fill=(255, 255, 255))

    y = 160
    lines = [
        "Generic Name: Liquid Dishwash Gel",
        "Net Quantity: 2 Litres",
        "MRP: Rs. 350.00 (inclusive of all taxes)",
        # Deliberately OMITTED: Unit Sale Price (required for >1L under GSR 779(E))
        "Month & Year of Manufacture: JUL 2026",
        "Manufactured by: CleanHome Products India",
        "Factory: Village Kheda, Dist Ahmedabad, Gujarat 382430",
        # Deliberately OMITTED: Consumer Care Phone / Email under Rule 6(1)(f)
    ]
    for line in lines:
        draw.text((60, y), line, fill=(20, 30, 50))
        y += 50

    draw.rectangle([(30, 30), (1070, 770)], outline=(100, 130, 180), width=3)
    img.save(path, "PNG")
    print(f"Created Demo 3 image: {path}")


def seed_all_demos():
    init_db()
    db = SessionLocal()
    try:
        admin = db.query(User).filter(User.username == "admin").first()
        if not admin:
            print("Run the server once to seed default users.")
            return

        demo_dir = os.path.abspath("./data/demo")
        os.makedirs(demo_dir, exist_ok=True)

        scenarios = [
            {
                "id": "LM-2026-DEMO01",
                "product": "Nutri-Delight Whole Wheat Biscuits 500g",
                "brand": "NutriBake",
                "category": ProductCategory.PACKAGED_FOOD,
                "label": "DEMO 1 — High Compliance Baseline (Packaged Food)",
                "notes": "Full compliance baseline check. All Rule 6 mandatory declarations clearly printed.",
                "img_func": create_demo_image_1_compliant_food,
                "filename": "demo1_food_compliant.png",
                "is_imported": False,
            },
            {
                "id": "LM-2026-DEMO02",
                "product": "Glow-Radiance Nourishing Face Serum 30ml",
                "brand": "Aura Cosmeceuticals",
                "category": ProductCategory.COSMETICS,
                "label": "DEMO 2 — Difficult Image Sample (Cosmetics Glare & Low Contrast)",
                "notes": "Curved cosmetic packaging with foil reflection glare and faint contact details.",
                "img_func": create_demo_image_2_glare_cosmetic,
                "filename": "demo2_cosmetic_glare.png",
                "is_imported": False,
            },
            {
                "id": "LM-2026-DEMO03",
                "product": "Sparkle Ultra Dishwash Gel 2L",
                "brand": "CleanHome",
                "category": ProductCategory.HOUSEHOLD_COMMODITY,
                "label": "DEMO 3 — Deliberately Altered Non-Compliant Sample (Household)",
                "notes": "Controlled non-compliance test: Missing Consumer Care contact (Rule 6(1)(f)) and USP (GSR 779(E)).",
                "img_func": create_demo_image_3_missing_declarations,
                "filename": "demo3_household_non_compliant.png",
                "is_imported": False,
            }
        ]

        for s in scenarios:
            img_path = os.path.join(demo_dir, s["filename"])
            s["img_func"](img_path)

            # Check if inspection already exists
            existing = db.query(Inspection).filter(Inspection.inspection_id == s["id"]).first()
            if existing:
                print(f"Inspection {s['id']} already exists in database. Updating...")
                db.delete(existing)
                db.commit()

            insp = Inspection(
                inspection_id=s["id"],
                inspector_id=admin.id,
                product_name=s["product"],
                brand_name=s["brand"],
                product_category=s["category"],
                status=InspectionStatus.DRAFT,
                is_demo=True,
                demo_label=s["label"],
                notes=s["notes"],
            )
            db.add(insp)
            db.commit()
            db.refresh(insp)

            # Save inspection image
            pil_img = Image.open(img_path).convert("RGB")
            quality = assess_quality(pil_img)
            img_record = InspectionImage(
                inspection_id=insp.id,
                image_type="original",
                file_path=img_path,
                file_name=s["filename"],
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

            # Run full AI pipeline
            evidence_dir = os.path.join(settings.upload_dir, "evidence", s["id"])
            result = run_inspection_pipeline(
                image_path=img_path,
                product_category=insp.product_category,
                inspection_id=s["id"],
                is_imported=s["is_imported"],
                evidence_save_dir=evidence_dir,
            )

            insp.status = result.status
            insp.overall_confidence = result.confidence.prototype_confidence_score if result.confidence else None
            insp.completed_at = datetime.utcnow()

            # Persist OCR results
            if result.ocr_result:
                for token in result.ocr_result.tokens:
                    db.add(OCRResultModel(
                        inspection_id=insp.id,
                        source_image_id=img_record.id,
                        ocr_pass=token.ocr_pass,
                        text=token.text,
                        confidence=token.confidence,
                        bbox_ymin=token.bbox[0] if token.bbox else None,
                        bbox_xmin=token.bbox[1] if token.bbox else None,
                        bbox_ymax=token.bbox[2] if token.bbox else None,
                        bbox_xmax=token.bbox[3] if token.bbox else None,
                        polygon_json=token.polygon,
                    ))

            # Persist declarations & rules
            for decl in result.declarations:
                decl_status = _map_decl_status(decl)
                db.add(Declaration(
                    inspection_id=insp.id,
                    field=decl.field,
                    status=decl_status,
                    extracted_value=decl.extracted_value,
                    raw_ocr_text=decl.raw_ocr_text,
                    normalized_value=decl.normalized_value,
                    extraction_confidence=decl.extraction_confidence,
                ))

            for rr in result.rule_results:
                db.add(RuleResult(
                    inspection_id=insp.id,
                    rule_id=rr.rule_id,
                    declaration_field=rr.declaration_field,
                    status=rr.status,
                    severity=rr.severity,
                    legal_reference=rr.legal_reference,
                    official_source_url=rr.official_source_url,
                    details=rr.details,
                ))

            for ev in result.evidence:
                db.add(EvidenceModel(
                    inspection_id=insp.id,
                    source_image_id=img_record.id,
                    declaration_field=ev.declaration_field,
                    crop_file_path=ev.crop_file_path,
                    bbox_json=ev.bbox,
                    ocr_text=ev.ocr_text,
                    confidence=ev.confidence,
                    rule_id=ev.rule_id,
                ))

            db.commit()
            print(f"Processed {s['id']}: Status={insp.status.value}, Confidence={insp.overall_confidence:.2f}")

        print("\nAll 3 demo scenarios seeded and analyzed successfully!")
    finally:
        db.close()


if __name__ == "__main__":
    seed_all_demos()

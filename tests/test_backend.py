"""
LEGALMET AI — Backend Tests
Tests for image quality, OCR provider, declaration extractor, and rule engine.
No real images are required for unit tests — they use mocks and synthetic data.
"""
import pytest
import numpy as np
from PIL import Image
import io
import sys
import os

# Ensure backend package is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.services.image_quality import assess_quality
from backend.services.preprocessing import preprocess
from backend.services.ocr import (
    OCRToken, MockOCRProvider, PaddleOCRProvider, create_ocr_provider, run_multi_pass_ocr
)
from backend.services.declaration_extractor import (
    MRPExtractor, NetQuantityExtractor, ConsumerCareExtractor,
    ManufacturerPackerExtractor, ExpiryDateExtractor, ManufactureDateExtractor,
    extract_all_declarations
)
from backend.services.rule_engine import (
    evaluate_rules, compute_overall_status, RuleEvalResult
)
from backend.services.confidence import compute_confidence
from backend.database.models import ProductCategory, DeclarationStatus


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_test_image(width=800, height=600, color=(200, 200, 200)) -> np.ndarray:
    img = np.full((height, width, 3), color, dtype=np.uint8)
    return img


def make_blurry_image() -> np.ndarray:
    import cv2
    img = make_test_image()
    return cv2.GaussianBlur(img, (51, 51), 30)


def make_low_contrast_image() -> np.ndarray:
    return np.full((600, 800, 3), 128, dtype=np.uint8)


# ── Image Quality Tests ───────────────────────────────────────────────────────

class TestImageQuality:
    def test_good_image_gives_good_recommendation(self):
        """A sharp, high-contrast image should return GOOD or ACCEPTABLE."""
        import cv2
        img = make_test_image(color=(240, 240, 240))
        # Add text-like structure to improve Laplacian variance
        cv2.putText(img, "MRP Rs. 120", (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)
        result = assess_quality(img)
        assert result.resolution_ok, "800x600 should pass resolution check"
        assert result.quality_score >= 0.0

    def test_small_image_fails_resolution(self):
        img = make_test_image(width=100, height=100)
        result = assess_quality(img)
        assert not result.resolution_ok

    def test_blurry_image_detected(self):
        img = make_blurry_image()
        result = assess_quality(img)
        assert result.blur_detected, "Heavily blurred image should be detected as blurry"

    def test_low_contrast_detected(self):
        img = make_low_contrast_image()
        result = assess_quality(img)
        assert not result.contrast_ok or result.contrast_score < 30

    def test_recommendation_field_is_valid(self):
        img = make_test_image()
        result = assess_quality(img)
        assert result.quality_recommendation in ("GOOD", "ACCEPTABLE", "POOR", "UNUSABLE")

    def test_returns_message(self):
        img = make_test_image()
        result = assess_quality(img)
        assert isinstance(result.message, str) and len(result.message) > 0


# ── Preprocessing Tests ────────────────────────────────────────────────────────

class TestPreprocessing:
    def test_preprocess_returns_variants(self):
        img = make_test_image()
        result = preprocess(img)
        assert result.original is not None
        assert result.upscaled is not None
        assert result.enhanced_clahe is not None
        assert result.high_contrast is not None
        assert result.contrast_enhanced is not None
        assert result.sharpened is not None
        assert len(result.variants) >= 5

    def test_variants_have_same_shape(self):
        img = make_test_image(width=640, height=480)
        result = preprocess(img)
        assert result.upscaled.shape == result.enhanced_clahe.shape
        assert result.upscaled.shape == result.high_contrast.shape
        assert result.upscaled.shape == result.sharpened.shape

    def test_low_res_image_auto_upscaled(self):
        img = make_test_image(width=500, height=337)
        result = preprocess(img)
        assert result.scale_factor >= 3.0
        assert result.upscaled_dims[1] >= 1400

    def test_accepts_pil_image(self):
        pil = Image.fromarray(make_test_image())
        result = preprocess(pil)
        assert result.original is not None



# ── Mock OCR Tests ────────────────────────────────────────────────────────────

class TestMockOCR:
    def _make_tokens(self):
        return [
            OCRToken("MRP Rs. 120 inclusive of all taxes", 0.95, [100, 50, 130, 400], None, "original"),
            OCRToken("Net Wt. 500g", 0.92, [150, 50, 180, 300], None, "original"),
            OCRToken("Mfg. by ABC Foods Pvt Ltd 400001", 0.88, [200, 50, 230, 400], None, "original"),
            OCRToken("Best Before: Jun 2025", 0.91, [250, 50, 280, 300], None, "original"),
            OCRToken("Consumer Care: 1800-123-4567 care@abcfoods.com", 0.89, [300, 50, 330, 400], None, "original"),
        ]

    def test_mock_provider_returns_configured_tokens(self):
        tokens = self._make_tokens()
        provider = MockOCRProvider(mock_tokens=tokens)
        provider.initialize()
        img = make_test_image()
        result = provider.extract_text(img, "original")
        assert len(result) == len(tokens)
        assert result[0].text == tokens[0].text

    def test_multi_pass_returns_deduplicated_result(self):
        tokens = self._make_tokens()
        provider = MockOCRProvider(mock_tokens=tokens)
        img = make_test_image()
        result = run_multi_pass_ocr(provider, img, img, img)
        assert result.full_text != ""
        assert result.mean_confidence > 0.0
        assert "original" in result.passes_used


class TestPaddleOCRProvider:
    def test_paddle_ocr_result_parsing(self):
        """Test that PaddleOCR raw polygon + (text, score) format parses into OCRTokens."""
        provider = PaddleOCRProvider(min_conf=0.30)
        
        # Simulate PaddleOCR engine output structure
        class MockPaddleEngine:
            def ocr(self, img, cls=True):
                return [[
                    [[[50, 100], [400, 100], [400, 130], [50, 130]], ("MRP Rs. 120 (incl. of all taxes)", 0.965)],
                    [[[50, 150], [300, 150], [300, 180], [50, 180]], ("Net Qty: 500 g", 0.921)],
                    [[[50, 200], [200, 200], [200, 220], [50, 220]], ("Low conf noise", 0.150)],  # should be filtered
                ]]
        
        provider._ocr = MockPaddleEngine()
        img = make_test_image()
        tokens = provider.extract_text(img, "enhanced_clahe")
        
        assert len(tokens) == 2, "Low confidence token (< 0.30) must be filtered out"
        assert tokens[0].text == "MRP Rs. 120 (incl. of all taxes)"
        assert tokens[0].confidence == 0.965
        assert tokens[0].bbox == [100, 50, 130, 400]  # [ymin, xmin, ymax, xmax]
        assert tokens[0].ocr_pass == "enhanced_clahe"
        assert tokens[1].text == "Net Qty: 500 g"

    def test_factory_creates_ocr_provider(self):
        """Test create_ocr_provider returns an instance implementing OCRProvider."""
        provider = create_ocr_provider("mock")
        assert provider is not None
        assert hasattr(provider, "extract_text")


# ── Declaration Extractor Tests ────────────────────────────────────────────────

class TestMRPExtractor:
    def _extract(self, text):
        extractor = MRPExtractor()
        token = OCRToken(text=text, confidence=0.95, bbox=[0, 0, 10, 100])
        return extractor.extract([token], text)

    def test_extracts_mrp_with_rupee_symbol(self):
        result = self._extract("MRP ₹120 inclusive of all taxes")
        assert result is not None
        assert result.extracted_value == "120"
        assert "incl. taxes" in result.normalized_value
        assert result.extraction_confidence == 1.0

    def test_extracts_mrp_rs_format(self):
        result = self._extract("M.R.P. Rs.85.50 incl. all taxes")
        assert result is not None
        assert result.extracted_value == "85.50"

    def test_mrp_without_tax_inclusion_has_lower_confidence(self):
        result = self._extract("MRP ₹200")
        assert result is not None
        assert result.extraction_confidence < 1.0

    def test_no_mrp_returns_none(self):
        result = self._extract("Net Weight 500g packed in India")
        assert result is None


class TestNetQuantityExtractor:
    def _extract(self, text):
        return NetQuantityExtractor().extract([], text)

    def test_extracts_grams(self):
        result = self._extract("Net Wt. 500g")
        assert result is not None
        assert "500" in result.extracted_value

    def test_extracts_ml(self):
        result = self._extract("Net Content: 250 ml")
        assert result is not None
        assert "250" in result.extracted_value

    def test_extracts_kg(self):
        result = self._extract("Net Weight: 1 kg")
        assert result is not None
        assert "1" in result.extracted_value


class TestConsumerCareExtractor:
    def _extract(self, text):
        return ConsumerCareExtractor().extract([], text)

    def test_detects_phone_and_email(self):
        result = self._extract("Consumer Care: 1800-123-4567 care@brand.com")
        assert result is not None
        assert result.extraction_confidence >= 0.9

    def test_detects_phone_only(self):
        result = self._extract("Helpline: 9876543210")
        assert result is not None

    def test_detects_email_only(self):
        result = self._extract("Email: support@company.in")
        assert result is not None

    def test_no_contact_returns_none(self):
        result = self._extract("MRP Rs 120 Net Wt 500g")
        assert result is None

    def test_excludes_fssai_lic_number_from_phone(self):
        """14-digit FSSAI License numbers must never be classified as consumer care phone."""
        result = self._extract("CONSUMER CARE CELL Email cs@parle.biz LIC: Ko : 100150260 10913022002253")
        assert result is not None
        assert "10913022002253" not in result.extracted_value
        assert "100150260" not in result.extracted_value
        assert "cs@parle.biz" in result.extracted_value

    def test_excludes_standalone_pin_code_from_phone(self):
        """6-digit postal PIN numbers must never be classified as consumer care phone."""
        result = self._extract("Write to Consumer Care Cell at Mumbai 400057")
        assert result is not None
        assert "400057" not in result.extracted_value


class TestManufactureDateExtractor:
    def _extract(self, text):
        token = OCRToken(text=text, confidence=0.88, bbox=[100, 200, 120, 350])
        return ManufactureDateExtractor().extract([token], text)

    def test_extracts_pkd_format(self):
        result = self._extract("PKD: 18/6/20 BATCH:RA L8A B")
        assert result is not None
        assert result.extracted_value == "18/6/20"
        assert result.raw_ocr_text == "PKD: 18/6/20 BATCH:RA L8A B"

    def test_extracts_mfd_month_year(self):
        result = self._extract("MFD: 08/2026")
        assert result is not None
        assert result.extracted_value == "08/2026"


class TestExpiryDateExtractor:
    def _extract(self, text):
        return ExpiryDateExtractor().extract([], text)

    def test_extracts_best_before(self):
        result = self._extract("Best Before: Jun 2025")
        assert result is not None
        assert "2025" in result.extracted_value

    def test_extracts_expiry(self):
        result = self._extract("Exp. 12/2024")
        assert result is not None

    def test_no_date_returns_none(self):
        result = self._extract("Manufactured in India")
        assert result is None


# ── Rule Engine Tests ──────────────────────────────────────────────────────────

class TestRuleEngine:
    def _make_decl(self, field, value, confidence):
        """Create a minimal mock declaration object."""
        class MockDecl:
            pass
        d = MockDecl()
        d.field = field
        d.extracted_value = value
        d.raw_ocr_text = value or ""
        d.normalized_value = value
        d.extraction_confidence = confidence
        return d

    def test_found_high_confidence_passes(self):
        decls = [
            self._make_decl("MRP", "120", 0.95),
            self._make_decl("NET_QUANTITY", "500g", 0.90),
            self._make_decl("MANUFACTURER_PACKER_IMPORTER", "ABC Foods Pvt Ltd, Mumbai 400001", 0.85),
            self._make_decl("COMMON_GENERIC_NAME", "Wheat Biscuits", 0.80),
            self._make_decl("MONTH_YEAR_OF_MANUFACTURE", "JAN 2024", 0.92),
            self._make_decl("BEST_BEFORE_EXPIRY_DATE", "JUN 2025", 0.91),
            self._make_decl("CONSUMER_CARE_DETAILS", "Phone: 1800-123-4567", 0.88),
        ]
        results = evaluate_rules(decls, ProductCategory.PACKAGED_FOOD, quality_score=0.90)
        found = [r for r in results if r.status == DeclarationStatus.FOUND]
        # At least MRP and Net Qty should be FOUND
        found_fields = {r.declaration_field for r in found}
        assert "MRP" in found_fields
        assert "NET_QUANTITY" in found_fields

    def test_missing_mandatory_field_on_good_image_is_not_found(self):
        """Missing mandatory declaration on a good quality image → NOT_FOUND (not MANUAL_REVIEW)."""
        decls = []  # No declarations extracted
        results = evaluate_rules(decls, ProductCategory.PACKAGED_FOOD, quality_score=0.88)
        mrp_result = next((r for r in results if r.declaration_field == "MRP"), None)
        assert mrp_result is not None
        assert mrp_result.status in (DeclarationStatus.NOT_FOUND, DeclarationStatus.MANUAL_REVIEW)

    def test_missing_field_on_poor_image_routes_to_manual_review(self):
        """Missing declaration on a poor quality image → MANUAL_REVIEW (not NOT_FOUND)."""
        decls = []
        results = evaluate_rules(decls, ProductCategory.PACKAGED_FOOD, quality_score=0.25)
        mrp_result = next((r for r in results if r.declaration_field == "MRP"), None)
        assert mrp_result is not None
        assert mrp_result.status == DeclarationStatus.MANUAL_REVIEW

    def test_expiry_date_not_applicable_for_household(self):
        """Expiry date is NOT_APPLICABLE for household commodities."""
        results = evaluate_rules([], ProductCategory.HOUSEHOLD_COMMODITY, quality_score=0.90)
        expiry_result = next(
            (r for r in results if r.declaration_field == "BEST_BEFORE_EXPIRY_DATE"), None
        )
        assert expiry_result is not None
        assert expiry_result.status == DeclarationStatus.NOT_APPLICABLE

    def test_country_of_origin_not_applicable_if_not_imported(self):
        results = evaluate_rules([], ProductCategory.PACKAGED_FOOD, quality_score=0.90, is_imported=False)
        country_result = next(
            (r for r in results if r.declaration_field == "COUNTRY_OF_ORIGIN"), None
        )
        assert country_result is not None
        assert country_result.status == DeclarationStatus.NOT_APPLICABLE

    def test_overall_status_pass_when_all_found(self):
        class MockRR:
            pass
        results = []
        for field in ["MRP", "NET_QUANTITY"]:
            r = MockRR()
            r.status = DeclarationStatus.FOUND
            r.severity = "HIGH"
            r.declaration_field = field
            results.append(r)
        assert compute_overall_status(results) == "PASS"

    def test_overall_status_non_compliance_on_high_not_found(self):
        class MockRR:
            pass
        r = MockRR()
        r.status = DeclarationStatus.NOT_FOUND
        r.severity = "HIGH"
        r.declaration_field = "MRP"
        assert compute_overall_status([r]) == "POTENTIAL_NON_COMPLIANCE"


# ── Confidence Tests ───────────────────────────────────────────────────────────

class TestConfidence:
    def test_high_quality_and_found_gives_high_confidence(self):
        class D:
            extracted_value = "120"
            extraction_confidence = 0.95

        class R:
            status = DeclarationStatus.FOUND

        result = compute_confidence(
            image_quality_score=0.90,
            ocr_mean_confidence=0.92,
            extracted_declarations=[D(), D()],
            rule_results=[R(), R()],
        )
        assert result.prototype_confidence_score >= 0.70
        assert result.label == "Prototype Confidence Score"

    def test_disclaimer_present(self):
        result = compute_confidence(0.5, 0.5, [], [])
        assert "not statistically calibrated" in result.disclaimer

    def test_score_clipped_to_unit_interval(self):
        result = compute_confidence(1.0, 1.0, [], [])
        assert 0.0 <= result.prototype_confidence_score <= 1.0


class TestUnitSalePriceExtractor:
    def test_extracts_usp_per_gram(self):
        from backend.services.declaration_extractor import UnitSalePriceExtractor
        extractor = UnitSalePriceExtractor()
        res = extractor.extract([], "Unit Sale Price: Rs. 0.24 / g")
        assert res is not None
        assert "0.24" in res.extracted_value

    def test_extracts_usp_per_kg(self):
        from backend.services.declaration_extractor import UnitSalePriceExtractor
        extractor = UnitSalePriceExtractor()
        res = extractor.extract([], "USP: Rs 45.00/kg")
        assert res is not None
        assert "45.00" in res.extracted_value


class TestCountryOfOriginExtractor:
    def test_extracts_country_standard(self):
        from backend.services.declaration_extractor import CountryOfOriginExtractor
        extractor = CountryOfOriginExtractor()
        res = extractor.extract([], "Country of Origin: India")
        assert res is not None
        assert "India" in res.extracted_value

    def test_extracts_country_ocr_typo(self):
        from backend.services.declaration_extractor import CountryOfOriginExtractor
        extractor = CountryOfOriginExtractor()
        res = extractor.extract([], "Countryot Origin: India")
        assert res is not None
        assert "India" in res.extracted_value


class TestManufactureDateFormats:
    def test_extracts_month_and_year_words(self):
        from backend.services.declaration_extractor import ManufactureDateExtractor
        extractor = ManufactureDateExtractor()
        res = extractor.extract([], "Month & Year of Manufacture: AUG 2026")
        assert res is not None
        assert "AUG 2026" in res.extracted_value


class TestLicenseExtractor:
    def test_extracts_fssai_license(self):
        from backend.services.declaration_extractor import LicenseExtractor
        extractor = LicenseExtractor()
        res = extractor.extract([], "FSSAI Lic. No. 10015043001129")
        assert res is not None
        assert "10015043001129" in res.extracted_value


class TestLowResolutionPackagedProductRegression:
    def test_resolution_detection_and_upscaling(self):
        import os
        from PIL import Image
        from backend.services.preprocessing import preprocess

        test_img_path = "data/test/britannia_marie_gold_500x337.webp"
        if not os.path.exists(test_img_path):
            return

        pil_img = Image.open(test_img_path).convert("RGB")
        prep = preprocess(pil_img)

        assert prep.original_dims == (337, 500)
        assert prep.scale_factor >= 3.0
        assert prep.upscaled_dims[1] >= 1500
        assert "clahe" in prep.variants
        assert "contrast_enhanced" in prep.variants
        assert "adaptive_threshold" in prep.variants
        assert "sharpened" in prep.variants

    def test_declarations_extracted_from_tokens(self):
        from backend.services.ocr import OCRToken
        from backend.services.declaration_extractor import extract_all_declarations

        sample_tokens = [
            OCRToken("73 9~", 0.90, [95, 238, 111, 275], None, "original"),
            OCRToken("10.C0", 0.85, [125, 312, 140, 347], None, "sharpened"),
            OCRToken("3s0.14per 9", 0.88, [135, 316, 150, 391], None, "sharpened"),
            OCRToken("Marie", 0.85, [137, 108, 163, 177], None, "contrast_enhanced"),
            OCRToken("GOLD", 0.99, [156, 110, 178, 163], None, "adaptive_threshold"),
            OCRToken("240324", 0.85, [146, 316, 160, 366], None, "adaptive_threshold"),
            OCRToken("fssai", 0.99, [204, 129, 235, 192], None, "original"),
            OCRToken("10015043001129", 0.95, [227, 140, 243, 214], None, "contrast_enhanced"),
            OCRToken("BRITANNIA", 0.95, [227, 342, 239, 402], None, "contrast_enhanced"),
            OCRToken("Ceusvett (sro Cell,", 0.80, [240, 212, 251, 275], None, "adaptive_threshold"),
        ]
        full_text = "\n".join(t.text for t in sample_tokens)
        decls = extract_all_declarations(sample_tokens, full_text)
        decl_dict = {d.field: d.extracted_value for d in decls}

        assert "NET_QUANTITY" in decl_dict
        assert "73" in decl_dict["NET_QUANTITY"]

        assert "MRP" in decl_dict
        assert "10" in decl_dict["MRP"]

        assert "UNIT_SALE_PRICE" in decl_dict
        assert "0.14" in decl_dict["UNIT_SALE_PRICE"]

        assert "MANUFACTURER_PACKER_IMPORTER" in decl_dict
        assert "Britannia" in decl_dict["MANUFACTURER_PACKER_IMPORTER"]

        assert "COMMON_GENERIC_NAME" in decl_dict
        assert "Marie Gold" in decl_dict["COMMON_GENERIC_NAME"]

        assert "MONTH_YEAR_OF_MANUFACTURE" in decl_dict

        assert "LICENSE_NUMBER" in decl_dict
        assert "10015043001129" in decl_dict["LICENSE_NUMBER"]

        assert "CONSUMER_CARE_DETAILS" in decl_dict



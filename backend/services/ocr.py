"""
LEGALMET AI — Pluggable OCR Provider Interface & Multi-Variant Fusion Engine

Architecture:
    OCRProvider (abstract)
    ├── EasyOCRProvider     (EasyOCR with angle & low-confidence candidate retention)
    ├── PaddleOCRProvider   (PaddleOCR DB/DB++ text detector & CRNN recognizer)
    └── MockOCRProvider     (Deterministic testing / offline unit tests)

Key Architectural Capabilities:
- Automatic upscaling mapping (transforms upscaled bounding boxes back to original coordinate space).
- Multi-variant OCR pass execution (original, grayscale, CLAHE, contrast enhanced, adaptive threshold, sharpened, dot-matrix).
- Spatial (IoU / coverage) and string-similarity (SequenceMatcher) deduplication.
- Preserves highest-confidence detection, bounding box, raw text, and source preprocessing pass.
- Retains low-confidence candidates for downstream statutory classification.
"""
from __future__ import annotations
import abc
import difflib
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any
from backend.config import settings


# ── Data contract ─────────────────────────────────────────────────────────────

@dataclass
class OCRToken:
    """Single text token returned by an OCR pass."""
    text: str
    confidence: float
    bbox: List[int]                     # [ymin, xmin, ymax, xmax] in original image coordinate space
    polygon: Optional[List[List[int]]] = field(default=None)  # Corner points in original image coordinate space
    ocr_pass: str = "original"          # Source pass (e.g. original, grayscale, clahe, contrast_enhanced, etc.)
    raw_text: Optional[str] = None      # Unmodified raw text output
    upscaled_bbox: Optional[List[int]] = field(default=None)  # Bounding box in upscaled space if upscaled


@dataclass
class MultiPassOCRResult:
    """Aggregated result across all OCR passes for a single image."""
    tokens: List[OCRToken]
    full_text: str                      # All deduplicated tokens concatenated (newline-separated)
    mean_confidence: float
    passes_used: List[str]
    all_raw_candidates: List[OCRToken] = field(default_factory=list)  # Raw candidates before deduplication


# ── Abstract Provider ─────────────────────────────────────────────────────────

class OCRProvider(abc.ABC):
    """
    Abstract base class for OCR providers.
    Any OCR engine must implement these methods.
    """

    @abc.abstractmethod
    def initialize(self) -> bool:
        """Initialize / load the underlying OCR engine. Returns True on success."""
        ...

    @abc.abstractmethod
    def extract_text(
        self,
        image_np: np.ndarray,
        pass_type: str = "original",
        min_confidence: Optional[float] = None,
    ) -> List[OCRToken]:
        """
        Run OCR on a numpy image array (H, W, 3 RGB).
        Returns a list of OCRToken objects with text, confidence, and bounding boxes.
        Retains candidate tokens with confidence >= min_confidence.
        """
        ...


# ── EasyOCR Provider ──────────────────────────────────────────────────────────

class EasyOCRProvider(OCRProvider):
    """OCR provider backed by EasyOCR with low-confidence candidate retention."""

    def __init__(self):
        self._reader: Any = None
        self._languages: List[str] = settings.ocr_languages.split(",")
        self._min_conf: float = min(0.20, settings.ocr_min_confidence)

    def initialize(self) -> bool:
        if self._reader is not None:
            return True
        try:
            import easyocr  # type: ignore[import-not-found,import-untyped]
            self._reader = easyocr.Reader(
                self._languages,
                gpu=False,
                verbose=False,
            )
            return True
        except Exception as e:
            print(f"[EasyOCRProvider] Initialization failed: {e}")
            return False

    def extract_text(
        self,
        image_np: np.ndarray,
        pass_type: str = "original",
        min_confidence: Optional[float] = None,
    ) -> List[OCRToken]:
        if self._reader is None:
            if not self.initialize():
                return []

        threshold = min_confidence if min_confidence is not None else self._min_conf

        try:
            results = self._reader.readtext(image_np, detail=1, paragraph=False)  # type: ignore
        except Exception as e:
            print(f"[EasyOCRProvider] OCR error: {e}")
            return []

        tokens: List[OCRToken] = []
        for (polygon, text, confidence) in results:
            score = float(confidence)
            if score < threshold:
                continue

            cleaned_text = str(text).strip()
            if not cleaned_text:
                continue

            xs = [int(p[0]) for p in polygon]
            ys = [int(p[1]) for p in polygon]
            bbox = [min(ys), min(xs), max(ys), max(xs)]
            tokens.append(OCRToken(
                text=cleaned_text,
                confidence=round(score, 4),
                bbox=bbox,
                polygon=[[int(p[0]), int(p[1])] for p in polygon],
                ocr_pass=pass_type,
                raw_text=cleaned_text,
            ))
        return tokens


# ── PaddleOCR Provider ────────────────────────────────────────────────────────

class PaddleOCRProvider(OCRProvider):
    """
    OCR provider backed by PaddleOCR.
    Supports deep text detection (DB/DB++), directional angle classification,
    and CRNN / SVTR text recognition models with high accuracy on packaged commodity labels.
    """

    def __init__(self, lang: str = "en", use_angle_cls: bool = True, min_conf: Optional[float] = None):
        self._ocr: Any = None
        self._lang: str = lang or settings.ocr_languages.split(",")[0]
        self._use_angle_cls: bool = use_angle_cls
        self._min_conf: float = min_conf if min_conf is not None else min(0.20, settings.ocr_min_confidence)

    def initialize(self) -> bool:
        if self._ocr is not None:
            return True
        try:
            from paddleocr import PaddleOCR  # type: ignore[import-not-found,import-untyped]
            try:
                self._ocr = PaddleOCR(
                    use_angle_cls=self._use_angle_cls,
                    lang=self._lang,
                    show_log=False,
                )
            except TypeError:
                self._ocr = PaddleOCR(
                    use_angle_cls=self._use_angle_cls,
                    lang=self._lang,
                )
            return True
        except ImportError:
            return False
        except Exception as e:
            print(f"[PaddleOCRProvider] Initialization notice: {e}")
            return False

    def extract_text(
        self,
        image_np: np.ndarray,
        pass_type: str = "original",
        min_confidence: Optional[float] = None,
    ) -> List[OCRToken]:
        if self._ocr is None:
            if not self.initialize():
                return []

        threshold = min_confidence if min_confidence is not None else self._min_conf

        try:
            results = self._ocr.ocr(image_np, cls=self._use_angle_cls)  # type: ignore
        except Exception as e:
            print(f"[PaddleOCRProvider] OCR extraction error: {e}")
            return []

        tokens: List[OCRToken] = []
        if not results:
            return tokens

        page_results = results[0] if results and isinstance(results[0], list) else results

        for item in page_results:
            if not item or len(item) < 2:
                continue
            polygon = item[0]          # [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
            text_score = item[1]        # (text, score) tuple
            if isinstance(text_score, (tuple, list)):
                text, score = text_score[0], float(text_score[1])
            else:
                text, score = str(text_score), 1.0

            if score < threshold:
                continue

            cleaned_text = str(text).strip()
            if not cleaned_text:
                continue

            xs = [int(p[0]) for p in polygon]
            ys = [int(p[1]) for p in polygon]
            bbox = [min(ys), min(xs), max(ys), max(xs)]

            tokens.append(OCRToken(
                text=cleaned_text,
                confidence=round(score, 4),
                bbox=bbox,
                polygon=[[int(p[0]), int(p[1])] for p in polygon],
                ocr_pass=pass_type,
                raw_text=cleaned_text,
            ))
        return tokens


# ── OCR Provider Factory ──────────────────────────────────────────────────────

def create_ocr_provider(provider_name: Optional[str] = None) -> OCRProvider:
    """
    Factory function to instantiate the configured or requested OCR provider.
    Fallback chain: requested provider → EasyOCR → Mock.
    """
    name = (provider_name or settings.ocr_provider or "auto").lower().strip()

    if name in ("paddle", "paddleocr"):
        provider = PaddleOCRProvider()
        if provider.initialize():
            return provider
        name = "easyocr"

    if name in ("easy", "easyocr", "auto"):
        provider = EasyOCRProvider()
        if provider.initialize():
            return provider

    if name == "auto":
        paddle_prov = PaddleOCRProvider()
        if paddle_prov.initialize():
            return paddle_prov

    mock = MockOCRProvider()
    mock.initialize()
    return mock


# ── Mock OCR Provider (testing) ───────────────────────────────────────────────

class MockOCRProvider(OCRProvider):
    """Deterministic mock OCR provider for unit and integration testing."""

    def __init__(self, mock_tokens: Optional[List[OCRToken]] = None):
        self._mock_tokens = mock_tokens or []

    def initialize(self) -> bool:
        return True

    def extract_text(
        self,
        image_np: np.ndarray,
        pass_type: str = "original",
        min_confidence: Optional[float] = None,
    ) -> List[OCRToken]:
        return [
            OCRToken(
                text=t.text,
                confidence=t.confidence,
                bbox=t.bbox,
                polygon=t.polygon,
                ocr_pass=pass_type,
                raw_text=t.text,
            )
            for t in self._mock_tokens
        ]


# ── Multi-Pass Runner ─────────────────────────────────────────────────────────

def run_multi_pass_ocr(
    provider: OCRProvider,
    original: Optional[np.ndarray] = None,
    enhanced_clahe: Optional[np.ndarray] = None,
    high_contrast: Optional[np.ndarray] = None,
    dot_matrix_enhanced: Optional[np.ndarray] = None,
    grayscale: Optional[np.ndarray] = None,
    contrast_enhanced: Optional[np.ndarray] = None,
    sharpened: Optional[np.ndarray] = None,
    adaptive_threshold: Optional[np.ndarray] = None,
    scale_factor: float = 1.0,
    preprocess_result: Any = None,
    variants_dict: Optional[Dict[str, np.ndarray]] = None,
) -> MultiPassOCRResult:
    """
    Runs multi-pass OCR on multiple image preprocessing variants:
    1. original / upscaled
    2. grayscale
    3. CLAHE enhanced
    4. contrast enhanced
    5. sharpened
    6. adaptive threshold / high_contrast
    7. dot_matrix_enhanced

    Maps bounding boxes back to the original image coordinate space,
    deduplicates overlapping text regions based on spatial overlap and text similarity,
    and preserves the highest-confidence candidate and its source pass.
    """
    passes: List[Tuple[np.ndarray, str]] = []

    if preprocess_result is not None:
        if hasattr(preprocess_result, "scale_factor"):
            scale_factor = float(preprocess_result.scale_factor)
        if hasattr(preprocess_result, "variants") and preprocess_result.variants:
            variants = preprocess_result.variants
            # Prioritize complementary feature enhancements (original, CLAHE, contrast, binarization)
            for v_name in ["original", "clahe", "contrast_enhanced", "adaptive_threshold"]:
                if v_name in variants and variants[v_name] is not None:
                    passes.append((variants[v_name], v_name))

    if not passes and variants_dict:
        for v_name, v_img in variants_dict.items():
            if v_img is not None:
                passes.append((v_img, v_name))

    if not passes:
        # Build passes from explicit arguments
        if original is not None:
            passes.append((original, "original"))
        if grayscale is not None:
            passes.append((grayscale, "grayscale"))
        if enhanced_clahe is not None:
            passes.append((enhanced_clahe, "clahe"))
        if contrast_enhanced is not None:
            passes.append((contrast_enhanced, "contrast_enhanced"))
        if sharpened is not None:
            passes.append((sharpened, "sharpened"))
        if adaptive_threshold is not None:
            passes.append((adaptive_threshold, "adaptive_threshold"))
        elif high_contrast is not None:
            passes.append((high_contrast, "high_contrast"))
        if dot_matrix_enhanced is not None:
            passes.append((dot_matrix_enhanced, "dot_matrix_enhanced"))

    all_raw_candidates: List[OCRToken] = []
    passes_used: List[str] = []

    for img, pass_name in passes:
        if img is None:
            continue
        passes_used.append(pass_name)
        tokens = provider.extract_text(img, pass_type=pass_name)

        # Rescale bboxes & polygons back to original image coordinates if image was upscaled
        for tok in tokens:
            tok.upscaled_bbox = list(tok.bbox)
            if abs(scale_factor - 1.0) > 0.02:
                ymin = int(round(tok.bbox[0] / scale_factor))
                xmin = int(round(tok.bbox[1] / scale_factor))
                ymax = int(round(tok.bbox[2] / scale_factor))
                xmax = int(round(tok.bbox[3] / scale_factor))
                tok.bbox = [ymin, xmin, ymax, xmax]
                if tok.polygon:
                    tok.polygon = [
                        [int(round(p[0] / scale_factor)), int(round(p[1] / scale_factor))]
                        for p in tok.polygon
                    ]
            all_raw_candidates.append(tok)

    # Deduplicate overlapping/near-identical text using bounding boxes and text similarity
    deduped = _deduplicate_tokens(all_raw_candidates)

    full_text = "\n".join(t.text for t in deduped if t.text.strip())
    mean_conf = (
        round(sum(t.confidence for t in deduped) / len(deduped), 4)
        if deduped else 0.0
    )

    return MultiPassOCRResult(
        tokens=deduped,
        full_text=full_text,
        mean_confidence=mean_conf,
        passes_used=passes_used,
        all_raw_candidates=all_raw_candidates,
    )


def _string_similarity(a: str, b: str) -> float:
    """Normalized character-level similarity ratio in range [0.0, 1.0]."""
    s1 = a.strip().lower()
    s2 = b.strip().lower()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
    return difflib.SequenceMatcher(None, s1, s2).ratio()


def _deduplicate_tokens(tokens: List[OCRToken]) -> List[OCRToken]:
    """
    Spatial & Text-Similarity Deduplication:
    1. Sort tokens by descending confidence.
    2. For candidate tokens, check spatial IoU / coverage and text similarity against accepted tokens.
    3. If two detections overlap in the same physical region or share near-identical text:
       - Keep the higher-confidence detection and its source pass.
       - If a candidate has richer / more complete text with comparable confidence, keep the richer text.
    4. Sort final tokens in natural reading order (top-to-bottom, left-to-right).
    """
    kept: List[OCRToken] = []

    # Sort primarily by descending confidence
    sorted_tokens = sorted(tokens, key=lambda t: -t.confidence)

    for token in sorted_tokens:
        is_dup = False
        for idx, existing in enumerate(kept):
            iou = _bbox_overlap(token.bbox, existing.bbox)
            cov = _bbox_coverage(token.bbox, existing.bbox)
            text_sim = _string_similarity(token.text, existing.text)

            # Spatial overlap match (same bounding box region)
            if iou > 0.25 or cov > 0.50 or (iou > 0.15 and text_sim > 0.55):
                is_dup = True
                if len(token.text) >= len(existing.text) + 3 and token.confidence >= 0.85 * existing.confidence:
                    kept[idx] = token
                break

            # Near-identical text with spatial proximity
            if text_sim > 0.85 and _spatial_distance(token.bbox, existing.bbox) < 40:
                is_dup = True
                break

        if not is_dup:
            kept.append(token)

    # Sort in natural reading order (row-wise top-to-bottom, then left-to-right)
    kept.sort(key=lambda t: (t.bbox[0] // 20, t.bbox[1]))
    return kept


def _spatial_distance(a: List[int], b: List[int]) -> float:
    """Euclidean distance between center points of two bboxes [ymin, xmin, ymax, xmax]."""
    cy_a = (a[0] + a[2]) / 2.0
    cx_a = (a[1] + a[3]) / 2.0
    cy_b = (b[0] + b[2]) / 2.0
    cx_b = (b[1] + b[3]) / 2.0
    return ((cy_a - cy_b) ** 2 + (cx_a - cx_b) ** 2) ** 0.5


def _bbox_overlap(a: List[int], b: List[int]) -> float:
    """Intersection-over-Union for two [ymin, xmin, ymax, xmax] bboxes."""
    iy1 = max(a[0], b[0])
    ix1 = max(a[1], b[1])
    iy2 = min(a[2], b[2])
    ix2 = min(a[3], b[3])
    intersection = max(0, iy2 - iy1) * max(0, ix2 - ix1)
    if intersection == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def _bbox_coverage(a: List[int], b: List[int]) -> float:
    """Percentage of box a's area contained in box b."""
    iy1 = max(a[0], b[0])
    ix1 = max(a[1], b[1])
    iy2 = min(a[2], b[2])
    ix2 = min(a[3], b[3])
    intersection = max(0, iy2 - iy1) * max(0, ix2 - ix1)
    if intersection == 0:
        return 0.0
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    return intersection / area_a if area_a > 0 else 0.0

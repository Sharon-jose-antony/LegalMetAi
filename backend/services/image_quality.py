"""
LEGALMET AI — Image Quality Assessment Service
Computes objective quality metrics from a PIL Image or numpy array.
All thresholds are loaded from config.py — no magic numbers here.
"""
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from backend.config import settings


@dataclass
class ImageQualityResult:
    quality_score: float
    resolution_ok: bool
    width_px: int
    height_px: int
    blur_score: float
    blur_detected: bool
    contrast_score: float
    contrast_ok: bool
    glare_fraction: float
    glare_detected: bool
    quality_recommendation: str  # GOOD | ACCEPTABLE | POOR | UNUSABLE
    message: str


def _to_numpy(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        return image
    return np.array(image.convert("RGB"))


def assess_quality(image: Image.Image | np.ndarray) -> ImageQualityResult:
    """
    Assess the quality of a package image for OCR suitability.

    Returns a structured result with individual metric scores and an
    overall quality_recommendation from: GOOD, ACCEPTABLE, POOR, UNUSABLE.

    The quality_score is a weighted composite in [0, 1].
    """
    img_np = _to_numpy(image)
    h, w = img_np.shape[:2]
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    # ── Resolution ────────────────────────────────────────────────────────────
    min_px = settings.iq_min_resolution_px
    resolution_ok = w >= min_px and h >= min_px

    # ── Blur — Laplacian Variance ──────────────────────────────────────────────
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    blur_detected = blur_score < settings.iq_blur_threshold

    # ── Contrast — Standard Deviation of Grayscale ────────────────────────────
    contrast_score = float(gray.std())
    contrast_ok = contrast_score >= settings.iq_contrast_threshold

    # ── Glare — Fraction of pixels with near-maximum brightness ───────────────
    bright_pixels = np.sum(gray > 245)
    glare_fraction = float(bright_pixels / gray.size)
    glare_detected = glare_fraction > settings.iq_glare_threshold

    # ── Composite Quality Score ────────────────────────────────────────────────
    # Normalize each component to [0, 1]; higher is better.
    blur_norm = min(blur_score / (settings.iq_blur_threshold * 4), 1.0)  # clamp at 4× threshold
    contrast_norm = min(contrast_score / (settings.iq_contrast_threshold * 4), 1.0)
    glare_norm = max(0.0, 1.0 - (glare_fraction / (settings.iq_glare_threshold * 4)))
    res_norm = 1.0 if resolution_ok else max(0.0, min(w, h) / min_px)

    quality_score = round(
        0.35 * blur_norm + 0.30 * contrast_norm + 0.20 * glare_norm + 0.15 * res_norm,
        4,
    )

    # ── Recommendation ────────────────────────────────────────────────────────
    issues = []
    if not resolution_ok:
        issues.append(f"resolution too low ({w}×{h}px, minimum {min_px}×{min_px}px)")
    if blur_detected:
        issues.append(f"image blurry (Laplacian variance {blur_score:.1f}, threshold {settings.iq_blur_threshold})")
    if not contrast_ok:
        issues.append(f"low contrast (std {contrast_score:.1f}, threshold {settings.iq_contrast_threshold})")
    if glare_detected:
        issues.append(f"glare detected ({glare_fraction*100:.1f}% bright pixels)")

    if not issues:
        recommendation = "GOOD"
        message = "Image quality is suitable for automated analysis."
    elif quality_score >= 0.55:
        recommendation = "ACCEPTABLE"
        message = "Image quality is marginal. Enhanced OCR passes will be applied. Declarations with low confidence will be routed to manual review. Issues: " + "; ".join(issues)
    elif quality_score >= 0.30:
        recommendation = "POOR"
        message = "Image quality is poor. OCR results may be unreliable. Manual inspection strongly recommended. Issues: " + "; ".join(issues)
    else:
        recommendation = "UNUSABLE"
        message = "Image quality is insufficient for automated analysis. Please capture a clearer image with better lighting and focus. Issues: " + "; ".join(issues)

    return ImageQualityResult(
        quality_score=quality_score,
        resolution_ok=resolution_ok,
        width_px=w,
        height_px=h,
        blur_score=round(blur_score, 2),
        blur_detected=blur_detected,
        contrast_score=round(contrast_score, 2),
        contrast_ok=contrast_ok,
        glare_fraction=round(glare_fraction, 4),
        glare_detected=glare_detected,
        quality_recommendation=recommendation,
        message=message,
    )

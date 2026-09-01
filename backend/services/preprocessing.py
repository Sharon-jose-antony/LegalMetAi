"""
LEGALMET AI — Image Preprocessing & Multi-Variant Enhancement Service
Applies resolution detection, automatic 3x–4x high-fidelity upscaling, and
OpenCV transformations to dramatically improve OCR accuracy on small, blurry,
glare-affected, and low-contrast package labels.

Generates multiple preprocessing variants:
1. original (upscaled RGB)
2. grayscale
3. CLAHE (Contrast Limited Adaptive Histogram Equalization)
4. contrast_enhanced (dynamic contrast stretching)
5. adaptive_threshold / high_contrast (Gaussian adaptive binarization)
6. sharpened (unsharp mask edge enhancement)
7. dot_matrix_enhanced (morphological stroke bridging for inkjet/date stamps)
"""
from __future__ import annotations
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class PreprocessResult:
    original: np.ndarray                 # Original raw input RGB array (unscaled)
    upscaled: np.ndarray                 # Base upscaled RGB array used for OCR passes
    scale_factor: float                  # Scale multiplier applied (1.0 = no upscale, 3.0-4.0 = upscaled)
    original_dims: Tuple[int, int]       # (height, width) of input image
    upscaled_dims: Tuple[int, int]       # (height, width) of upscaled image
    grayscale: np.ndarray                # Grayscale in 3-channel RGB
    enhanced_clahe: np.ndarray           # CLAHE contrast enhancement + unsharp mask
    contrast_enhanced: np.ndarray        # Contrast stretched / scaled RGB
    adaptive_threshold: np.ndarray       # Adaptive threshold binarization in 3-channel RGB
    high_contrast: np.ndarray            # Adaptive threshold / high-contrast binarization (backward compatible)
    sharpened: np.ndarray                # Unsharp masked sharpened RGB
    dot_matrix_enhanced: np.ndarray      # Morphological closing + stroke sharpening for inkjet/date stamps
    variants: Dict[str, np.ndarray] = field(default_factory=dict)


def _to_numpy_rgb(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, np.ndarray):
        if len(image.shape) == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif len(image.shape) == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image
    return np.array(image.convert("RGB"))


def detect_and_upscale(
    img_rgb: np.ndarray,
    target_long_edge: int = 1600,
    min_scale: float = 0.25,
    max_scale: float = 4.0,
) -> Tuple[np.ndarray, float]:
    """
    Detects input image resolution:
    - Low-resolution images (<1000px): Automatically upscales 3x–4x using Lanczos4 interpolation.
    - Ultra-high resolution photos from phone cameras (>1920px, e.g. 4000x3000):
      Downscales with antialiased Lanczos4 interpolation for fast OCR.
    - Normal resolution (1200px - 1920px): Preserved as-is (1.0x).

    Parameters:
        img_rgb: Input RGB image as numpy array (H, W, 3).
        target_long_edge: Desired long dimension in pixels for optimal OCR text recognition.
        min_scale: Minimum scale multiplier.
        max_scale: Maximum scale multiplier.

    Returns:
        (normalized_rgb, scale_factor)
    """
    h, w = img_rgb.shape[:2]
    long_edge = max(h, w)
    short_edge = min(h, w)

    # Determine adaptive scaling factor
    if long_edge < 800 or short_edge < 500:
        # Low resolution (e.g. 500x337): upscale 3x - 4x for crystal clear small font recognition
        scale = max(3.0, min(max_scale, target_long_edge / float(long_edge)))
    elif long_edge < 1200 or short_edge < 800:
        # Medium-low resolution: upscale 1.4x - 2.0x
        scale = max(1.4, min(2.5, target_long_edge / float(long_edge)))
    elif long_edge > 1920:
        # High-res mobile camera photo (e.g. 4032x3024): downscale to ~1600px for speed
        scale = max(min_scale, target_long_edge / float(long_edge))
    else:
        # Optimal resolution: 1.0 (no scaling needed)
        scale = 1.0

    scale = round(scale, 3)

    if abs(scale - 1.0) > 0.02:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))
        # Lanczos4 provides superior antialiased edge preservation
        normalized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        return normalized, scale

    return img_rgb, 1.0


def preprocess(image: Image.Image | np.ndarray) -> PreprocessResult:
    """
    Produce enhanced multi-pass variants for OCR with resolution detection and upscaling:
    1. Original / Upscaled RGB
    2. Grayscale
    3. CLAHE Enhanced
    4. Contrast Enhanced
    5. Adaptive Threshold (high_contrast)
    6. Sharpened (Unsharp Mask)
    7. Dot-Matrix / Inkjet Enhanced
    """
    original = _to_numpy_rgb(image)
    orig_h, orig_w = original.shape[:2]

    # Detect resolution and automatically upscale low-resolution images (3x–4x)
    upscaled, scale_factor = detect_and_upscale(original)
    up_h, up_w = upscaled.shape[:2]

    # 1. Grayscale
    gray = cv2.cvtColor(upscaled, cv2.COLOR_RGB2GRAY)
    gray_rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization) + High-Definition Unsharp Mask
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    clahe_gray = clahe.apply(gray)
    blurred_clahe = cv2.GaussianBlur(clahe_gray, (0, 0), 2.0)
    sharpened_clahe = cv2.addWeighted(clahe_gray, 1.8, blurred_clahe, -0.8, 0)
    enhanced_clahe_rgb = cv2.cvtColor(sharpened_clahe, cv2.COLOR_GRAY2RGB)

    # 3. Contrast Enhanced (Linear stretch + brightness normalization)
    contrast_gray = cv2.convertScaleAbs(gray, alpha=1.35, beta=-15)
    contrast_rgb = cv2.cvtColor(contrast_gray, cv2.COLOR_GRAY2RGB)

    # 4. Adaptive Thresholding (Gaussian C binarization for glare / reflective print)
    block_size = 21 if scale_factor >= 2.0 else 15
    adapt_thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, 10
    )
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    adapt_clean = cv2.morphologyEx(adapt_thresh, cv2.MORPH_OPEN, kernel_clean)
    adapt_rgb = cv2.cvtColor(adapt_clean, cv2.COLOR_GRAY2RGB)

    # 5. Sharpened (Direct Unsharp Masking)
    blur_sharp = cv2.GaussianBlur(gray, (0, 0), 3.0)
    sharp_gray = cv2.addWeighted(gray, 1.85, blur_sharp, -0.85, 0)
    sharpened_rgb = cv2.cvtColor(sharp_gray, cv2.COLOR_GRAY2RGB)

    # 6. Dot-Matrix / Inkjet Enhanced (Morphological closing + high-pass stroke filter)
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(clahe_gray, cv2.MORPH_CLOSE, kernel_close)
    blur_close = cv2.GaussianBlur(closed, (0, 0), 1.5)
    dot_matrix_sharp = cv2.addWeighted(closed, 1.9, blur_close, -0.9, 0)
    dot_matrix_rgb = cv2.cvtColor(dot_matrix_sharp, cv2.COLOR_GRAY2RGB)

    variants = {
        "original": upscaled,
        "grayscale": gray_rgb,
        "clahe": enhanced_clahe_rgb,
        "contrast_enhanced": contrast_rgb,
        "adaptive_threshold": adapt_rgb,
        "sharpened": sharpened_rgb,
        "dot_matrix_enhanced": dot_matrix_rgb,
    }

    return PreprocessResult(
        original=original,
        upscaled=upscaled,
        scale_factor=scale_factor,
        original_dims=(orig_h, orig_w),
        upscaled_dims=(up_h, up_w),
        grayscale=gray_rgb,
        enhanced_clahe=enhanced_clahe_rgb,
        contrast_enhanced=contrast_rgb,
        adaptive_threshold=adapt_rgb,
        high_contrast=adapt_rgb,
        sharpened=sharpened_rgb,
        dot_matrix_enhanced=dot_matrix_rgb,
        variants=variants,
    )


def crop_region(image: np.ndarray, bbox: list[int], padding: int = 10) -> np.ndarray:
    """
    Crop a bounding-box region from an image with optional padding.
    bbox format: [ymin, xmin, ymax, xmax]
    """
    h, w = image.shape[:2]
    ymin = max(0, bbox[0] - padding)
    xmin = max(0, bbox[1] - padding)
    ymax = min(h, bbox[2] + padding)
    xmax = min(w, bbox[3] + padding)
    return image[ymin:ymax, xmin:xmax]


def draw_bbox_highlight(image: np.ndarray, bbox: list[int], label: str = "",
                        color: tuple = (0, 120, 255), thickness: int = 2) -> np.ndarray:
    """
    Draw a coloured bounding box with an optional label on a copy of the image.
    Returns the annotated image without mutating the input.
    """
    img_copy = image.copy()
    ymin, xmin, ymax, xmax = bbox
    cv2.rectangle(img_copy, (xmin, ymin), (xmax, ymax), color, thickness)
    if label:
        cv2.putText(
            img_copy, label,
            (xmin, max(ymin - 5, 12)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )
    return img_copy

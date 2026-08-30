"""
LEGALMET AI — Evidence Extraction Service
Generates cropped image evidence slices with annotated bounding boxes
for each detected declaration.
Every declaration result must be traceable back to the raw image evidence.
"""
from __future__ import annotations
import os
import cv2
import numpy as np
from PIL import Image
from dataclasses import dataclass
from typing import List, Optional
from backend.services.ocr import OCRToken
from backend.services.preprocessing import crop_region, draw_bbox_highlight
from backend.config import settings


@dataclass
class EvidenceSlice:
    declaration_field: str
    crop_image: Optional[np.ndarray]   # Cropped + annotated image region
    crop_file_path: Optional[str]      # Saved to disk for persistence
    bbox: Optional[List[int]]          # [ymin, xmin, ymax, xmax]
    ocr_text: str
    confidence: float
    rule_id: Optional[str]


def generate_evidence(
    tokens: List[OCRToken],
    original_image: np.ndarray,
    declaration_field: str,
    matched_text: str,
    rule_id: Optional[str] = None,
    inspection_id: str = "unknown",
    save_dir: Optional[str] = None,
) -> Optional[EvidenceSlice]:
    """
    Find the OCR token(s) matching the extracted declaration text,
    crop that region from the image, draw a highlight box, and save to disk.
    """
    # Find the best matching token by text overlap
    best_token: Optional[OCRToken] = None
    best_score = 0.0
    match_lower = matched_text.lower().strip()

    for token in tokens:
        token_lower = token.text.lower().strip()
        # Jaccard-like overlap on words
        match_words = set(match_lower.split())
        token_words = set(token_lower.split())
        if not match_words or not token_words:
            continue
        overlap = len(match_words & token_words) / len(match_words | token_words)
        if overlap > best_score:
            best_score = overlap
            best_token = token

    if best_token is None or best_score < 0.20:
        # No matching token found — return evidence with no crop
        return EvidenceSlice(
            declaration_field=declaration_field,
            crop_image=None,
            crop_file_path=None,
            bbox=None,
            ocr_text=matched_text,
            confidence=0.0,
            rule_id=rule_id,
        )

    bbox = best_token.bbox  # [ymin, xmin, ymax, xmax]
    crop = crop_region(original_image, bbox, padding=12)
    annotated = draw_bbox_highlight(
        original_image, bbox,
        label=declaration_field,
        color=(0, 120, 255)
    )

    # Save crop image
    file_path = None
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        safe_field = declaration_field.replace("/", "_").replace(" ", "_")
        filename = f"{inspection_id}_{safe_field}_evidence.jpg"
        file_path = os.path.join(save_dir, filename)
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, crop_rgb)

    return EvidenceSlice(
        declaration_field=declaration_field,
        crop_image=crop,
        crop_file_path=file_path,
        bbox=bbox,
        ocr_text=best_token.text,
        confidence=best_token.confidence,
        rule_id=rule_id,
    )

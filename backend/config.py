"""
LEGALMET AI — Application Configuration
All thresholds and settings are loaded from environment variables.
No magic numbers are scattered through service code.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import List
import os


class Settings(BaseSettings):
    # ── App ──────────────────────────────────────────────────────────────────
    app_name: str = "LEGALMET AI"
    app_env: str = "development"
    debug: bool = True
    secret_key: str = "CHANGE_ME_IN_PRODUCTION_MIN_32_CHARS_STRONG_KEY"

    # ── Database ──────────────────────────────────────────────────────────────
    database_url: str = "sqlite:///./legalmet_ai.db"

    # ── File Uploads ──────────────────────────────────────────────────────────
    upload_dir: str = "./data/uploads"
    max_file_size_mb: int = 20
    allowed_extensions: str = "jpg,jpeg,png,webp,bmp,tiff"

    # ── Image Quality Thresholds ─────────────────────────────────────────────
    iq_min_resolution_px: int = 400      # Minimum width AND height in pixels
    iq_blur_threshold: float = 80.0      # Laplacian variance; below = blurry
    iq_contrast_threshold: float = 30.0  # Std-dev of grayscale; below = low contrast
    iq_glare_threshold: float = 0.08     # Fraction of very bright pixels; above = glare

    # ── OCR ───────────────────────────────────────────────────────────────────
    ocr_provider: str = "paddleocr"      # "paddleocr" (primary) | "easyocr" | "auto" | "mock"
    ocr_languages: str = "en"
    ocr_min_confidence: float = 0.30

    # ── Confidence Weights ────────────────────────────────────────────────────
    conf_weight_quality: float = 0.20
    conf_weight_ocr: float = 0.35
    conf_weight_extraction: float = 0.35
    conf_weight_applicability: float = 0.10

    # ── Report ────────────────────────────────────────────────────────────────
    reports_dir: str = "./data/reports"

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: str = "http://localhost:3000,http://localhost:8080,http://127.0.0.1:3000"

    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip().lower() for ext in self.allowed_extensions.split(",")]

    @property
    def cors_origins_list(self) -> List[str]:
        return [o.strip() for o in self.cors_origins.split(",")]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )


settings = Settings()

# Ensure upload and report directories exist at startup
os.makedirs(settings.upload_dir, exist_ok=True)
os.makedirs(settings.reports_dir, exist_ok=True)
os.makedirs("./data/demo", exist_ok=True)
os.makedirs("./data/test", exist_ok=True)

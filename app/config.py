"""Configuration module for the application."""

import os


class Config:
    """Application configuration class."""

    DIS_BASE_URL = os.getenv("DIS_BASE_URL", "http://34.142.158.178")
    FLASK_ENV = os.getenv("FLASK_ENV", "production")

    TRANSFORMERS_CACHE = "/app/.cache/transformers"
    HF_HOME = "/app/.cache/huggingface"

    @classmethod
    def get_endpoints(cls):
        """Get service endpoints configuration."""
        return {
            "acquisition": f"{cls.DIS_BASE_URL}/api/acquisition",
            "ingestion": f"{cls.DIS_BASE_URL}/api/ingestion",
            "processing": f"{cls.DIS_BASE_URL}/api/processing",
        }

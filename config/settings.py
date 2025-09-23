"""
Simple Settings Configuration
Centralized environment variable management for FastAPI server
"""

import os
from typing import Optional


class Settings:
    """Simple settings class for environment variables"""

    def __init__(self):
        # AWS S3 Configuration
        self.s3_bucket: str = os.getenv("S3_BUCKET_NAME", os.getenv("S3_BUCKET", ""))

        # Callback URLs
        self.fastapi_base_url: str = os.getenv("FASTAPI_BASE_URL", "")  # https://ho-it.site
        self.backend_url: str = os.getenv("BACKEND_URL", "")  # Fargate 개발 URL

        # Derived settings
        self.default_callback_url: str = self.fastapi_base_url or self.backend_url
        self.enable_callbacks: bool = bool(self.default_callback_url and self.default_callback_url.strip())

    def get_callback_url(self, override_url: Optional[str] = None) -> str:
        """Get callback URL with optional override"""
        return override_url or self.default_callback_url


# Global settings instance
settings = Settings()
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Akinator API"
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: str = "https://api.z.ai/api/coding/paas/v4"
    LLM_MODEL_NAME: str = "glm-5-turbo"
    LLM_TIMEOUT_SECONDS: float = 60.0

    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    model_config = SettingsConfigDict(
        env_file=BACKEND_ROOT / ".env",
        env_file_encoding="utf-8-sig",
        extra="ignore",
    )


settings = Settings()

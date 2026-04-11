from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    PROJECT_NAME: str = "AI Akinator API"
    LLM_API_KEY: Optional[str] = None
    LLM_BASE_URL: str = "https://api.groq.com/openai/v1"
    LLM_MODEL_NAME: str = "llama-3.3-70b-versatile"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8-sig"

settings = Settings()

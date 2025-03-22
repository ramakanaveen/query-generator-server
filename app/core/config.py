import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Debug mode
    DEBUG: bool = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
    
    # API Keys
    API_KEY_ANTHROPIC: str = os.getenv("API_KEY_ANTHROPIC", "")
    
    # Google settings
    GOOGLE_CREDENTIALS_PATH: str = os.getenv("GOOGLE_CREDENTIALS_PATH", "./google-credentials.json")
    GOOGLE_PROJECT_ID: str = os.getenv("GOOGLE_PROJECT_ID", "")
    GOOGLE_LOCATION: str = os.getenv("GOOGLE_LOCATION", "us-central1")
    
    # Gemini model settings
    GEMINI_MODEL_NAME: str = os.getenv("GEMINI_MODEL_NAME", "gemini-1.5-pro-002")
    GEMINI_TEMPERATURE: float = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
    GEMINI_TOP_P: float = float(os.getenv("GEMINI_TOP_P", "0.95"))
    
    # Claude model settings
    CLAUDE_MODEL_NAME: str = os.getenv("CLAUDE_MODEL_NAME", "claude-3-sonnet-20240229")
    CLAUDE_TEMPERATURE: float = float(os.getenv("CLAUDE_TEMPERATURE", "0.2"))
    CLAUDE_MAX_TOKENS: int = int(os.getenv("CLAUDE_MAX_TOKENS", "4000"))
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./test.db")
    
    # KDB settings
    KDB_HOST: str = os.getenv("KDB_HOST", "localhost")
    KDB_PORT: int = int(os.getenv("KDB_PORT", "5000"))
    
    # CORS settings
    CORS_ORIGINS: List[str] = [
        "http://localhost:3000",  # React development server
        "http://localhost:8000",  # For local testing
        "https://your-production-domain.com",
    ]
    
    class Config:
        case_sensitive = True

settings = Settings()
import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

class Settings(BaseSettings):
    PROJECT_NAME: str = "TalentForge v3 — AI Job Discovery & Application Intelligence"
    API_V1_STR: str = "/api/v1"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite+aiosqlite:///./talentforge_v3.db")
    
    # API Keys for Free-Tier Multi-LLM Router
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GOOGLE_AI_STUDIO_API_KEY: str = os.getenv("GOOGLE_AI_STUDIO_API_KEY", "")
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    GITHUB_MODELS_TOKEN: str = os.getenv("GITHUB_MODELS_TOKEN", "")
    HUGGINGFACE_API_TOKEN: str = os.getenv("HUGGINGFACE_API_TOKEN", "")
    
    # Sourcing APIs
    ADZUNA_APP_ID: str = os.getenv("ADZUNA_APP_ID", "sample_app_id")
    ADZUNA_APP_KEY: str = os.getenv("ADZUNA_APP_KEY", "sample_app_key")
    
    # Email Digest
    RESEND_API_KEY: str = os.getenv("RESEND_API_KEY", "")
    
    # Observability
    LANGCHAIN_TRACING_V2: bool = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"
    LANGCHAIN_API_KEY: str = os.getenv("LANGCHAIN_API_KEY", "")
    
    class Config:
        case_sensitive = True
        env_file = ".env"
        extra = "ignore"

settings = Settings()

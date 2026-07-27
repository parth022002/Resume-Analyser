import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Load environment variables from .env file
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env")
load_dotenv(dotenv_path=env_path)

class Settings(BaseModel):
    PROJECT_NAME: str = "TalentForge AI Career Intelligence Platform"
    VERSION: str = "2.0.0"
    API_V1_STR: str = "/api/v1"
    
    # Environment
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    
    # Neon PostgreSQL Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "postgresql://neondb_owner:npg_lrHvxfj6d5Ba@ep-bold-dream-aztdx0m2-pooler.c-3.ap-southeast-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
    )
    
    # Provider API Keys
    OPENROUTER_API_KEY: str = os.getenv("OPENROUTER_API_KEY", "")
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    
    # Multi-Model Tier Routing Configuration (Supports Gemini, DeepSeek, NVIDIA Nemotron, Gemma, Llama)
    TIER_0_MODEL: str = os.getenv("TIER_0_MODEL", "openrouter/google/gemini-2.5-flash:free")
    TIER_1_MODEL: str = os.getenv("TIER_1_MODEL", "openrouter/deepseek/deepseek-chat")
    TIER_2_MODEL: str = os.getenv("TIER_2_MODEL", "openrouter/nvidia/nemotron-4-340b-instruct")
    
    # Security / CORS
    CORS_ORIGINS: list = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://talentforge.vercel.app"
    ]

settings = Settings()

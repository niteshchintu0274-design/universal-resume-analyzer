import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")


def _as_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(value: str | None, default: int) -> int:
    try:
        return int(value) if value is not None else default
    except ValueError:
        return default


def _as_float(value: str | None, default: float) -> float:
    try:
        return float(value) if value is not None else default
    except ValueError:
        return default


class Config:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-change-this-secret-key")
    DEBUG = _as_bool(os.getenv("FLASK_DEBUG"), True)

    BASE_DIR = BASE_DIR
    UPLOAD_FOLDER = Path(os.getenv("UPLOAD_FOLDER", BASE_DIR / "uploads"))
    DATABASE_PATH = Path(os.getenv("DATABASE_PATH", BASE_DIR / "database" / "resume_analyzer.db"))

    MAX_CONTENT_LENGTH = _as_int(os.getenv("MAX_CONTENT_LENGTH"), 8 * 1024 * 1024)
    ALLOWED_EXTENSIONS = {"pdf"}

    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
    CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-20250514")
    CLAUDE_MAX_TOKENS = _as_int(os.getenv("CLAUDE_MAX_TOKENS"), 1200)
    CLAUDE_TEMPERATURE = _as_float(os.getenv("CLAUDE_TEMPERATURE"), 0.2)
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    GEMINI_MAX_TOKENS = _as_int(os.getenv("GEMINI_MAX_TOKENS"), 1200)
    GEMINI_TEMPERATURE = _as_float(os.getenv("GEMINI_TEMPERATURE"), 0.2)

    RAG_PROVIDER = os.getenv("RAG_PROVIDER", "local").lower()
    CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", BASE_DIR / "database" / "chroma"))

    SPACY_MODEL = os.getenv("SPACY_MODEL", "en_core_web_sm")
    MIN_RESUME_TEXT_LENGTH = _as_int(os.getenv("MIN_RESUME_TEXT_LENGTH"), 80)

    TAILWIND_CDN_URL = os.getenv("TAILWIND_CDN_URL", "https://cdn.tailwindcss.com")

    JSEARCH_RAPIDAPI_KEY = os.getenv("JSEARCH_RAPIDAPI_KEY") or os.getenv("RAPIDAPI_KEY")
    ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
    ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
    ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "us")
    RAPIDAPI_JOBS_KEY = os.getenv("RAPIDAPI_JOBS_KEY")
    RAPIDAPI_JOBS_HOST = os.getenv("RAPIDAPI_JOBS_HOST")
    RAPIDAPI_JOBS_URL = os.getenv("RAPIDAPI_JOBS_URL")
    LINKEDIN_JOBS_API_URL = os.getenv("LINKEDIN_JOBS_API_URL")
    LINKEDIN_JOBS_API_TOKEN = os.getenv("LINKEDIN_JOBS_API_TOKEN")


class DevelopmentConfig(Config):
    DEBUG = True


class ProductionConfig(Config):
    DEBUG = False


config_by_name = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "default": DevelopmentConfig,
}

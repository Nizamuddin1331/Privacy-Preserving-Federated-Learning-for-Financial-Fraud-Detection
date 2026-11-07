"""Application settings (environment variables / .env)."""
from functools import lru_cache
from pathlib import Path
from typing import List

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "Privacy-Preserving FL for Fraud Detection"
    database_url: str = f"sqlite:///{PROJECT_ROOT / 'fl_fraud.db'}"   # use postgresql+psycopg://... in prod
    results_dir: str = str(PROJECT_ROOT / "results")
    data_dir: str = str(PROJECT_ROOT / "data")
    reports_dir: str = str(PROJECT_ROOT / "reports")
    cors_origins: List[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    log_level: str = "INFO"


@lru_cache
def get_settings() -> Settings:
    return Settings()

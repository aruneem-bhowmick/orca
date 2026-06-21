"""Application settings loaded from environment variables."""

from __future__ import annotations

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """BFF configuration – all values can be overridden via env vars."""

    # Database
    database_url: str = "postgresql+asyncpg://orca:orca_dev_secret@localhost:5432/orca_registry"

    # Redis
    redis_url: str = "redis://localhost:6379"

    # JWT
    jwt_secret_key: str = "dev-secret-change-in-prod"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 15
    refresh_token_expire_days: int = 7

    # OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    github_client_id: str = ""
    github_client_secret: str = ""

    # Upstream services
    orcamind_api_url: str = "http://localhost:8000"
    orcalab_api_url: str = "http://localhost:8001"
    orcanet_api_url: str = "http://localhost:8002"

    # CORS / frontend
    frontend_url: str = "http://localhost:5173"
    cors_origins: str = "http://localhost:5173,http://localhost:3000"

    model_config = {"env_prefix": "", "case_sensitive": False}


settings = Settings()

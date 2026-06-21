"""Tests for orca_web.config."""

import os

from orca_web.config import Settings


class TestSettings:
    def test_defaults(self):
        s = Settings()
        assert s.jwt_algorithm == "HS256"
        assert s.access_token_expire_minutes == 15
        assert s.refresh_token_expire_days == 7
        assert s.orcamind_api_url == "http://localhost:8000"
        assert s.orcalab_api_url == "http://localhost:8001"
        assert s.orcanet_api_url == "http://localhost:8002"
        assert s.frontend_url == "http://localhost:5173"
        assert s.google_client_id == ""
        assert s.github_client_id == ""

    def test_env_override(self, monkeypatch):
        monkeypatch.setenv("JWT_SECRET_KEY", "from-env")
        monkeypatch.setenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30")
        monkeypatch.setenv("ORCAMIND_API_URL", "http://mind:9000")
        s = Settings()
        assert s.jwt_secret_key == "from-env"
        assert s.access_token_expire_minutes == 30
        assert s.orcamind_api_url == "http://mind:9000"

    def test_cors_origins_default(self):
        s = Settings()
        assert "localhost:5173" in s.cors_origins
        assert "localhost:3000" in s.cors_origins

    def test_database_url_default(self):
        s = Settings()
        assert "asyncpg" in s.database_url
        assert "orca_registry" in s.database_url

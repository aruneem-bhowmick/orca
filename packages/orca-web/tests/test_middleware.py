"""Tests for orca_web.api.middleware."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI

from orca_web.api.middleware import add_middleware


@pytest.fixture(autouse=True)
def _patch_settings(monkeypatch):
    s = SimpleNamespace(
        cors_origins="http://localhost:5173,http://localhost:3000",
    )
    monkeypatch.setattr("orca_web.api.middleware.settings", s)


class TestAddMiddleware:
    def test_attaches_cors_and_logging(self):
        app = FastAPI()
        initial_count = len(app.user_middleware)
        add_middleware(app)
        assert len(app.user_middleware) > initial_count

    def test_empty_cors_origins_allows_all(self, monkeypatch):
        monkeypatch.setattr(
            "orca_web.api.middleware.settings",
            SimpleNamespace(cors_origins=""),
        )
        app = FastAPI()
        add_middleware(app)
        # Middleware was added without raising
        assert len(app.user_middleware) >= 1

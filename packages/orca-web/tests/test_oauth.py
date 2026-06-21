"""Tests for orca_web.auth.oauth module-level client registration."""

from __future__ import annotations

import importlib
import sys
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from orca_web.auth.oauth import oauth


def _reload_oauth_with(settings, mock_oauth_instance):
    """Remove the cached oauth module, patch settings and OAuth, then re-import."""
    saved = sys.modules.pop("orca_web.auth.oauth", None)
    try:
        with patch("orca_web.config.settings", settings), \
             patch("authlib.integrations.starlette_client.OAuth",
                   return_value=mock_oauth_instance):
            mod = importlib.import_module("orca_web.auth.oauth")
        return mod
    finally:
        # Restore original module so other tests are unaffected
        if saved is not None:
            sys.modules["orca_web.auth.oauth"] = saved


class TestOAuthRegistration:
    def test_default_settings_produce_oauth_instance(self):
        """The module exports an OAuth instance regardless of credentials."""
        from authlib.integrations.starlette_client import OAuth
        assert isinstance(oauth, OAuth)

    def test_google_registered_when_credentials_set(self):
        """Google client is registered when google_client_id is non-empty."""
        settings = SimpleNamespace(
            google_client_id="goog-id",
            google_client_secret="goog-secret",
            github_client_id="",
            github_client_secret="",
        )
        mock_oauth_instance = MagicMock()
        _reload_oauth_with(settings, mock_oauth_instance)

        calls = mock_oauth_instance.register.call_args_list
        names = [c.kwargs.get("name") for c in calls]
        assert "google" in names
        assert "github" not in names

    def test_github_registered_when_credentials_set(self):
        """GitHub client is registered when github_client_id is non-empty."""
        settings = SimpleNamespace(
            google_client_id="",
            google_client_secret="",
            github_client_id="gh-id",
            github_client_secret="gh-secret",
        )
        mock_oauth_instance = MagicMock()
        _reload_oauth_with(settings, mock_oauth_instance)

        calls = mock_oauth_instance.register.call_args_list
        names = [c.kwargs.get("name") for c in calls]
        assert "github" in names
        assert "google" not in names

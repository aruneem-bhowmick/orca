"""Unit tests verifying that orcalab.api.main exports a module-level app instance."""

from __future__ import annotations

import fastapi
import pytest

import orcalab.api.main as main_module


class TestAppExport:
    def test_app_attribute_exists(self) -> None:
        assert hasattr(main_module, "app")

    def test_app_is_fastapi_instance(self) -> None:
        assert isinstance(main_module.app, fastapi.FastAPI)

    def test_app_title(self) -> None:
        assert main_module.app.title == "OrcaLab"


class TestAppRoutes:
    @pytest.fixture(scope="class")
    def route_paths(self) -> set[str]:
        return {route.path for route in main_module.app.routes}

    def test_health_route_registered(self, route_paths: set[str]) -> None:
        assert "/health" in route_paths

    def test_root_route_registered(self, route_paths: set[str]) -> None:
        assert "/" in route_paths

    def test_sweeps_routes_registered(self, route_paths: set[str]) -> None:
        assert any("/api/v1/sweeps" in p for p in route_paths)

    def test_experiments_routes_registered(self, route_paths: set[str]) -> None:
        assert any("/api/v1/experiments" in p for p in route_paths)

    def test_search_spaces_routes_registered(self, route_paths: set[str]) -> None:
        assert any("/api/v1/search-spaces" in p for p in route_paths)

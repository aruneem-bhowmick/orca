"""Validate pyproject.toml files for both the workspace root and each package."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def root_toml(repo_root: Path) -> dict:
    return tomllib.loads((repo_root / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def orcamind_toml(orcamind_pkg_dir: Path) -> dict:
    return tomllib.loads((orcamind_pkg_dir / "pyproject.toml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def orca_shared_toml(repo_root: Path) -> dict:
    path = repo_root / "packages" / "orca-shared" / "pyproject.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


# ── Root pyproject.toml ───────────────────────────────────────────────────────

def test_root_pyproject_exists(repo_root: Path) -> None:
    assert (repo_root / "pyproject.toml").is_file()


def test_root_has_no_project_section(root_toml: dict) -> None:
    assert "project" not in root_toml, (
        "Root pyproject.toml should be a workspace coordinator only, not an installable package"
    )


def test_root_uv_workspace_defined(root_toml: dict) -> None:
    assert "tool" in root_toml
    assert "uv" in root_toml["tool"]
    assert "workspace" in root_toml["tool"]["uv"]


def test_root_workspace_members_count(root_toml: dict) -> None:
    members = root_toml["tool"]["uv"]["workspace"]["members"]
    assert len(members) == 2


def test_root_workspace_includes_orca_shared(root_toml: dict) -> None:
    members = root_toml["tool"]["uv"]["workspace"]["members"]
    assert "packages/orca-shared" in members


def test_root_workspace_includes_orcamind(root_toml: dict) -> None:
    members = root_toml["tool"]["uv"]["workspace"]["members"]
    assert "packages/orcamind" in members


def test_root_ruff_line_length(root_toml: dict) -> None:
    assert root_toml["tool"]["ruff"]["line-length"] == 100


def test_root_ruff_target_version(root_toml: dict) -> None:
    assert root_toml["tool"]["ruff"]["target-version"] == "py311"


def test_root_ruff_select_contains_essentials(root_toml: dict) -> None:
    select = root_toml["tool"]["ruff"]["select"]
    for code in ("E", "F", "I"):
        assert code in select, f"Ruff rule set '{code}' missing from select"


def test_root_mypy_python_version(root_toml: dict) -> None:
    assert root_toml["tool"]["mypy"]["python_version"] == "3.11"


def test_root_mypy_overrides_orca_shared_strict(root_toml: dict) -> None:
    overrides = root_toml["tool"]["mypy"]["overrides"]
    orca_shared_override = next(
        (o for o in overrides if o.get("module", "").startswith("orca_shared")), None
    )
    assert orca_shared_override is not None, "No mypy override found for orca_shared.*"
    assert orca_shared_override["strict"] is True


def test_root_pytest_testpaths(root_toml: dict) -> None:
    paths = root_toml["tool"]["pytest.ini_options"]["testpaths"]
    assert "packages/orca-shared/tests" in paths
    assert "packages/orcamind/tests" in paths


def test_root_pytest_asyncio_mode(root_toml: dict) -> None:
    assert root_toml["tool"]["pytest.ini_options"]["asyncio_mode"] == "auto"


# ── orcamind pyproject.toml ───────────────────────────────────────────────────

def test_orcamind_pyproject_exists(orcamind_pkg_dir: Path) -> None:
    assert (orcamind_pkg_dir / "pyproject.toml").is_file()


def test_orcamind_name(orcamind_toml: dict) -> None:
    assert orcamind_toml["project"]["name"] == "orcamind"


def test_orcamind_version(orcamind_toml: dict) -> None:
    assert orcamind_toml["project"]["version"] == "0.1.0"


def test_orcamind_requires_python_311(orcamind_toml: dict) -> None:
    assert "3.11" in orcamind_toml["project"]["requires-python"]


def test_orcamind_build_backend_is_hatchling(orcamind_toml: dict) -> None:
    assert orcamind_toml["build-system"]["build-backend"] == "hatchling.build"


def test_orcamind_script_entry_point(orcamind_toml: dict) -> None:
    scripts = orcamind_toml["project"]["scripts"]
    assert scripts.get("orcamind") == "orcamind.cli:app"


REQUIRED_DEPS = [
    "torch",
    "torchvision",
    "pytorch-lightning",
    "learn2learn",
    "higher",
    "numpy",
    "scipy",
    "scikit-learn",
    "pandas",
    "xgboost",
    "mlflow",
    "hydra-core",
    "omegaconf",
    "faiss-cpu",
    "fastapi",
    "uvicorn",
    "pydantic",
    "httpx",
    "sqlalchemy",
    "asyncpg",
    "redis",
    "minio",
    "streamlit",
    "plotly",
    "typer",
    "orca-shared",
]


@pytest.mark.parametrize("pkg", REQUIRED_DEPS)
def test_orcamind_dependency_present(pkg: str, orcamind_toml: dict) -> None:
    deps: list[str] = orcamind_toml["project"]["dependencies"]
    matches = [d for d in deps if d.lower().startswith(pkg.lower())]
    assert matches, f"Dependency '{pkg}' not found in orcamind's [project.dependencies]"


def test_orcamind_dev_extras_include_pytest(orcamind_toml: dict) -> None:
    dev = orcamind_toml["project"]["optional-dependencies"]["dev"]
    assert any(d.startswith("pytest>=") for d in dev)


def test_orcamind_dev_extras_include_ruff(orcamind_toml: dict) -> None:
    dev = orcamind_toml["project"]["optional-dependencies"]["dev"]
    assert any(d.startswith("ruff>=") for d in dev)


def test_orcamind_dev_extras_include_mypy(orcamind_toml: dict) -> None:
    dev = orcamind_toml["project"]["optional-dependencies"]["dev"]
    assert any(d.startswith("mypy>=") for d in dev)


# ── orca-shared pyproject.toml ────────────────────────────────────────────────

def test_orca_shared_pyproject_exists(repo_root: Path) -> None:
    assert (repo_root / "packages" / "orca-shared" / "pyproject.toml").is_file()


def test_orca_shared_name(orca_shared_toml: dict) -> None:
    assert orca_shared_toml["project"]["name"] == "orca-shared"


def test_orca_shared_version(orca_shared_toml: dict) -> None:
    assert orca_shared_toml["project"]["version"] == "0.1.0"


def test_orca_shared_build_backend(orca_shared_toml: dict) -> None:
    assert orca_shared_toml["build-system"]["build-backend"] == "hatchling.build"


def test_orca_shared_wheel_package_name(orca_shared_toml: dict) -> None:
    packages = orca_shared_toml["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
    assert "orca_shared" in packages


def test_orca_shared_requires_python_311(orca_shared_toml: dict) -> None:
    assert "3.11" in orca_shared_toml["project"]["requires-python"]

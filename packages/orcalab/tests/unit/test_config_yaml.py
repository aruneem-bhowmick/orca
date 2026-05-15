"""Validate the Hydra configuration files for structure and expected values."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def root_config(config_dir: Path) -> dict:
    return yaml.safe_load((config_dir / "config.yaml").read_text())


@pytest.fixture(scope="module")
def bayesian_config(config_dir: Path) -> dict:
    return yaml.safe_load((config_dir / "search" / "bayesian.yaml").read_text())


@pytest.fixture(scope="module")
def asha_config(config_dir: Path) -> dict:
    return yaml.safe_load((config_dir / "pruner" / "asha.yaml").read_text())


# ── config.yaml ───────────────────────────────────────────────────────────────


def test_root_config_file_exists(config_dir: Path) -> None:
    assert (config_dir / "config.yaml").is_file()


def test_root_config_has_defaults(root_config: dict) -> None:
    assert "defaults" in root_config


def test_root_config_defaults_includes_self(root_config: dict) -> None:
    assert "_self_" in root_config["defaults"]


def test_root_config_has_prefect_section(root_config: dict) -> None:
    assert "prefect" in root_config


def test_root_config_prefect_api_url_contains_env_var(root_config: dict) -> None:
    assert "PREFECT_API_URL" in root_config["prefect"]["api_url"]


def test_root_config_prefect_work_pool(root_config: dict) -> None:
    assert root_config["prefect"]["work_pool"] == "orcalab-pool"


def test_root_config_has_orcamind_section(root_config: dict) -> None:
    assert "orcamind" in root_config


def test_root_config_orcamind_enabled_is_true(root_config: dict) -> None:
    assert root_config["orcamind"]["enabled"] is True


def test_root_config_orcamind_api_url_contains_env_var(root_config: dict) -> None:
    assert "ORCAMIND_API_URL" in root_config["orcamind"]["api_url"]


def test_root_config_has_resources_section(root_config: dict) -> None:
    assert "resources" in root_config


def test_root_config_max_parallel_experiments(root_config: dict) -> None:
    assert root_config["resources"]["max_parallel_experiments"] == 4


def test_root_config_gpu_per_experiment(root_config: dict) -> None:
    assert root_config["resources"]["gpu_per_experiment"] == 1


def test_root_config_timeout_seconds(root_config: dict) -> None:
    assert root_config["resources"]["timeout_seconds"] == 3600


# ── search/bayesian.yaml ──────────────────────────────────────────────────────


def test_bayesian_config_file_exists(config_dir: Path) -> None:
    assert (config_dir / "search" / "bayesian.yaml").is_file()


def test_bayesian_sampler_is_tpe(bayesian_config: dict) -> None:
    assert bayesian_config["sampler"] == "TPE"


def test_bayesian_n_startup_trials(bayesian_config: dict) -> None:
    assert bayesian_config["n_startup_trials"] == 10


def test_bayesian_n_ei_candidates(bayesian_config: dict) -> None:
    assert bayesian_config["n_ei_candidates"] == 24


def test_bayesian_multivariate_is_true(bayesian_config: dict) -> None:
    assert bayesian_config["multivariate"] is True


# ── pruner/asha.yaml ──────────────────────────────────────────────────────────


def test_asha_config_file_exists(config_dir: Path) -> None:
    assert (config_dir / "pruner" / "asha.yaml").is_file()


def test_asha_pruner_name(asha_config: dict) -> None:
    assert asha_config["pruner"] == "ASHA"


def test_asha_min_resource(asha_config: dict) -> None:
    assert asha_config["min_resource"] == 1


def test_asha_max_resource(asha_config: dict) -> None:
    assert asha_config["max_resource"] == 100


def test_asha_reduction_factor(asha_config: dict) -> None:
    assert asha_config["reduction_factor"] == 3

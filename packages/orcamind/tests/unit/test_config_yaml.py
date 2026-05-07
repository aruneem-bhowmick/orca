"""Validate the Hydra configuration files for structure and key values."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


@pytest.fixture(scope="module")
def root_config(config_dir: Path) -> dict:
    return yaml.safe_load((config_dir / "config.yaml").read_text())


@pytest.fixture(scope="module")
def maml_config(config_dir: Path) -> dict:
    return yaml.safe_load((config_dir / "model" / "maml.yaml").read_text())


@pytest.fixture(scope="module")
def openml_config(config_dir: Path) -> dict:
    return yaml.safe_load((config_dir / "dataset" / "openml.yaml").read_text())


@pytest.fixture(scope="module")
def adam_config(config_dir: Path) -> dict:
    return yaml.safe_load((config_dir / "optimizer" / "adam.yaml").read_text())


# ── config.yaml ───────────────────────────────────────────────────────────────

def test_root_config_file_exists(config_dir: Path) -> None:
    assert (config_dir / "config.yaml").is_file()


def test_root_config_has_defaults(root_config: dict) -> None:
    assert "defaults" in root_config


def test_root_config_defaults_contains_model(root_config: dict) -> None:
    defaults = root_config["defaults"]
    assert any(
        (isinstance(d, dict) and "model" in d) or d == {"model": "maml"}
        for d in defaults
    ) or any(d == "model: maml" or (isinstance(d, dict) and d.get("model") == "maml") for d in defaults)


def test_root_config_defaults_includes_self(root_config: dict) -> None:
    assert "_self_" in root_config["defaults"]


def test_root_config_has_paths(root_config: dict) -> None:
    assert "paths" in root_config


def test_root_config_paths_has_data_dir(root_config: dict) -> None:
    assert "data_dir" in root_config["paths"]


def test_root_config_paths_has_model_dir(root_config: dict) -> None:
    assert "model_dir" in root_config["paths"]


def test_root_config_paths_has_log_dir(root_config: dict) -> None:
    assert "log_dir" in root_config["paths"]


def test_root_config_paths_use_env_interpolation(root_config: dict) -> None:
    assert "ORCA_DATA_DIR" in root_config["paths"]["data_dir"]
    assert "ORCA_MODEL_DIR" in root_config["paths"]["model_dir"]


def test_root_config_has_mlflow_section(root_config: dict) -> None:
    assert "mlflow" in root_config


def test_root_config_mlflow_tracking_uri(root_config: dict) -> None:
    assert "MLFLOW_TRACKING_URI" in root_config["mlflow"]["tracking_uri"]


def test_root_config_mlflow_experiment_name(root_config: dict) -> None:
    assert root_config["mlflow"]["experiment_name"] == "orcamind"


def test_root_config_seed(root_config: dict) -> None:
    assert root_config["seed"] == 42


def test_root_config_device(root_config: dict) -> None:
    assert root_config["device"] == "cuda"


# ── model/maml.yaml ───────────────────────────────────────────────────────────

def test_maml_config_file_exists(config_dir: Path) -> None:
    assert (config_dir / "model" / "maml.yaml").is_file()


def test_maml_name(maml_config: dict) -> None:
    assert maml_config["name"] == "maml"


def test_maml_algorithm(maml_config: dict) -> None:
    assert maml_config["algorithm"] == "MAML"


def test_maml_inner_lr(maml_config: dict) -> None:
    assert maml_config["inner_lr"] == pytest.approx(0.01)


def test_maml_inner_steps(maml_config: dict) -> None:
    assert maml_config["inner_steps"] == 5


def test_maml_outer_lr(maml_config: dict) -> None:
    assert maml_config["outer_lr"] == pytest.approx(0.001)


def test_maml_meta_batch_size(maml_config: dict) -> None:
    assert maml_config["meta_batch_size"] == 4


def test_maml_base_model_type(maml_config: dict) -> None:
    assert maml_config["base_model"]["type"] == "mlp"


def test_maml_base_model_hidden_dims(maml_config: dict) -> None:
    assert maml_config["base_model"]["hidden_dims"] == [256, 128, 64]


def test_maml_max_meta_epochs(maml_config: dict) -> None:
    assert maml_config["max_meta_epochs"] == 1000


def test_maml_support_size(maml_config: dict) -> None:
    assert maml_config["support_size"] == 10


def test_maml_query_size(maml_config: dict) -> None:
    assert maml_config["query_size"] == 15


def test_maml_first_order_is_false(maml_config: dict) -> None:
    assert maml_config["first_order"] is False


# ── dataset/openml.yaml ───────────────────────────────────────────────────────

def test_openml_config_file_exists(config_dir: Path) -> None:
    assert (config_dir / "dataset" / "openml.yaml").is_file()


def test_openml_name(openml_config: dict) -> None:
    assert openml_config["name"] == "openml"


def test_openml_suite_id(openml_config: dict) -> None:
    assert openml_config["suite_id"] == 271


def test_openml_max_tasks(openml_config: dict) -> None:
    assert openml_config["max_tasks"] == 72


def test_openml_task_types_includes_classification(openml_config: dict) -> None:
    assert "classification" in openml_config["task_types"]


def test_openml_task_types_includes_regression(openml_config: dict) -> None:
    assert "regression" in openml_config["task_types"]


def test_openml_min_samples(openml_config: dict) -> None:
    assert openml_config["min_samples"] == 100


def test_openml_max_samples(openml_config: dict) -> None:
    assert openml_config["max_samples"] == 100000


def test_openml_train_ratio(openml_config: dict) -> None:
    assert openml_config["train_ratio"] == pytest.approx(0.7)


def test_openml_split_ratios_sum_to_one(openml_config: dict) -> None:
    total = openml_config["train_ratio"] + openml_config["val_ratio"] + openml_config["test_ratio"]
    assert total == pytest.approx(1.0)


def test_openml_cache_dir_references_paths(openml_config: dict) -> None:
    assert "paths.data_dir" in openml_config["cache_dir"]


def test_openml_force_reload_is_false(openml_config: dict) -> None:
    assert openml_config["force_reload"] is False


# ── optimizer/adam.yaml ───────────────────────────────────────────────────────

def test_adam_config_file_exists(config_dir: Path) -> None:
    assert (config_dir / "optimizer" / "adam.yaml").is_file()


def test_adam_name(adam_config: dict) -> None:
    assert adam_config["name"] == "adam"


def test_adam_type(adam_config: dict) -> None:
    assert adam_config["type"] == "Adam"


def test_adam_lr(adam_config: dict) -> None:
    assert adam_config["lr"] == pytest.approx(0.001)


def test_adam_betas(adam_config: dict) -> None:
    assert adam_config["betas"] == pytest.approx([0.9, 0.999])


def test_adam_weight_decay(adam_config: dict) -> None:
    assert adam_config["weight_decay"] == pytest.approx(1e-4)


def test_adam_amsgrad_is_false(adam_config: dict) -> None:
    assert adam_config["amsgrad"] is False


def test_adam_scheduler_type(adam_config: dict) -> None:
    assert adam_config["scheduler"]["type"] == "CosineAnnealingLR"


def test_adam_scheduler_tmax_matches_model_epochs(adam_config: dict) -> None:
    assert adam_config["scheduler"]["T_max"] == 1000


def test_adam_grad_clip_norm(adam_config: dict) -> None:
    assert adam_config["grad_clip_norm"] == pytest.approx(1.0)

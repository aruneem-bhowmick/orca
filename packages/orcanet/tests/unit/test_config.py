"""Tests that verify orcanet Hydra/OmegaConf config files are valid and well-formed."""

from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

CONFIG_DIR = Path(__file__).parents[3] / "config"


def _load(relative: str):
    """Load a config file relative to packages/orcanet/config/."""
    path = CONFIG_DIR / relative
    assert path.exists(), f"Config file missing: {path}"
    return OmegaConf.load(path)


def test_root_config_loads() -> None:
    """config.yaml parses without error."""
    cfg = _load("config.yaml")
    assert cfg is not None


def test_root_config_has_required_keys() -> None:
    """Root config contains the expected top-level keys."""
    cfg = _load("config.yaml")
    assert "llm" in cfg
    assert "retrieval" in cfg
    assert "orcamind" in cfg
    assert "orcalab" in cfg


def test_retrieval_config_values() -> None:
    """Retrieval config has sensible defaults for top_k and threshold."""
    cfg = _load("config.yaml")
    assert cfg.retrieval.top_k_initial == 50
    assert cfg.retrieval.top_k_final == 10
    assert cfg.retrieval.similarity_threshold == pytest.approx(0.6)
    assert cfg.retrieval.use_llm_reranking is True


def test_llm_config_defaults() -> None:
    """LLM config defaults to openai provider and gpt-4-turbo model."""
    cfg = _load("config.yaml")
    assert cfg.llm.provider == "openai"
    assert cfg.llm.model == "gpt-4-turbo"
    assert cfg.llm.temperature == pytest.approx(0.7)


def test_hybrid_retriever_config_loads() -> None:
    """config/retriever/hybrid.yaml parses without error."""
    cfg = _load("retriever/hybrid.yaml")
    assert cfg is not None


def test_hybrid_retriever_has_top_k() -> None:
    """Hybrid retriever config exposes top_k_initial and top_k_final."""
    cfg = _load("retriever/hybrid.yaml")
    assert "top_k_initial" in cfg
    assert "top_k_final" in cfg


def test_cross_domain_embedder_config_loads() -> None:
    """config/embedder/cross_domain.yaml parses without error."""
    cfg = _load("embedder/cross_domain.yaml")
    assert cfg is not None


def test_cross_domain_embedder_dims() -> None:
    """Embedder config has correct input/output dimensions."""
    cfg = _load("embedder/cross_domain.yaml")
    assert cfg.input_dim == 25
    assert cfg.embedding_dim == 64
    assert cfg.n_domains == 10
    assert cfg.n_task_types == 3


def test_openai_llm_config_loads() -> None:
    """config/llm/openai.yaml parses without error."""
    cfg = _load("llm/openai.yaml")
    assert cfg is not None


def test_openai_llm_config_has_model() -> None:
    """OpenAI LLM config specifies provider, model, and max_tokens."""
    cfg = _load("llm/openai.yaml")
    assert cfg.provider == "openai"
    assert cfg.model == "gpt-4-turbo"
    assert cfg.max_tokens == 2048


def test_all_config_files_are_valid_yaml() -> None:
    """Every .yaml file under config/ is parseable by OmegaConf."""
    for yaml_file in CONFIG_DIR.rglob("*.yaml"):
        cfg = OmegaConf.load(yaml_file)
        assert cfg is not None, f"Failed to parse {yaml_file}"

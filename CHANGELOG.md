# Changelog

All notable changes to the Orca project are documented in this file,
derived from the merge-request history of the repository.

## Foundation — Monorepo Scaffold and Shared Infrastructure

**PRs #1–3 · May 7–8, 2026**

### Added

- Initialised uv-managed monorepo with ruff, mypy, pytest, and pre-commit
  tooling (#1).
- Scaffolded `orcamind` package with Hydra configuration hierarchy, module
  stubs, CLI entry-point, multi-stage Dockerfile, and docker-compose
  development environment (#1).
- Scaffolded `orca-shared` package with TaskRepository, OrcaMindClient,
  OrcaLabClient, centralised config/logging, and Pydantic domain models (#2).
- Added CODEOWNERS to enforce review gates on `main` (#2).
- Implemented `StatisticalTaskEmbedder` with 25-dimensional meta-feature
  extraction for dataset profiling (#3).

## OrcaMind Core — Meta-Learning Algorithms and Model Selection

**PRs #4–9 · May 9–11, 2026**

### Added

- Implemented `NeuralTaskEmbedder` with FAISS-backed similarity search for
  embedding-space nearest-neighbour retrieval (#4).
- Implemented MAML (Model-Agnostic Meta-Learning) algorithm with inner/outer
  loop optimisation and full test suite (#5).
- Added Reptile and Meta-SGD meta-learners as lightweight MAML alternatives
  (#6).
- Implemented `WarmStartTransfer` for checkpoint-based model warm-starting
  (#7).
- Built model selection framework with nearest-neighbour, ranking, and
  performance-prediction selectors (#8).
- Implemented meta-training pipeline with PyTorch Lightning integration for
  end-to-end learner training (#9).

## OrcaMind Services — Dataset Bootstrap, API, CLI, and Dashboard

**PRs #10–13 · May 12–13, 2026**

### Added

- Bootstrapped OrcaMind meta-dataset from OpenML benchmarks for real-world
  task coverage (#10).
- Implemented OrcaMind FastAPI service with 11 REST endpoints for model
  registry, embeddings, and recommendations (#11).
- Implemented all six OrcaMind CLI commands (`train`, `recommend`, `embed`,
  `list`, `evaluate`, `serve`) with full test suite (#12).
- Built Streamlit analytics dashboard with four pages and a performance
  summary API (#13).

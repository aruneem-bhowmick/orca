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

## OrcaMind Polish — Test Coverage, Docker, and Documentation

**PRs #14–19 · May 14, 2026**

### Added

- Comprehensive OrcaMind test coverage: Spearman correlation, embed schema,
  and full pipeline tests (#14).
- Real FAISS integration tests and fixes for pre-existing test failures (#15).
- Docker deployment with Alembic migrations and database initialisation for
  OrcaMind (#16).
- Revamped root README to reflect the current codebase (#17).
- Split monolithic README into a navigable `docs/` suite with architecture,
  component, getting-started, and development guides (#18).
- OrcaMind HTTP response shape assertion tests for integration endpoints
  (#19).

## OrcaLab Foundations — Package Scaffold and Search Strategies

**PRs #20–26 · May 15–16, 2026**

### Added

- Scaffolded `orcalab` package with module skeleton, CLI, config, and test
  suite (#20).
- Updated reference documentation for the new OrcaLab package (#21).
- Implemented composable search-space definitions with categorical,
  integer, float, and log-uniform parameter types (#22).
- Implemented `RandomSearch` and `GridSearch` strategies (#23).
- Added `BayesianSearch` backed by Optuna TPE with prior injection and
  persistence (#24).
- Added `MetaInformedSearch` strategy with OrcaMind warm-start and feedback
  loop (#25).
- Added `EvolutionarySearch` strategy backed by CMA-ES (#26).

## OrcaLab Features — Pruning, Experiment Lifecycle, Orchestration, and API

**PRs #27–33 · May 16–17, 2026**

### Added

- Implemented trial pruning strategies: ASHA, median stopping, and
  OrcaMind-informed meta-pruner (#27).
- CodeRabbit-generated unit tests for pruning module (#28).
- Implemented experiment lifecycle, `ExperimentRunner`, and
  `BatchRunner` for managed trial execution (#29).
- Added Prefect workflow layer for experiment orchestration with
  task-level retries and concurrency control (#30).
- Built OrcaLab Streamlit live dashboard for real-time experiment
  monitoring (#31).
- Implemented OrcaLab REST and WebSocket API service (#32).
- Implemented OrcaMind bidirectional integration: OrcaLab queries OrcaMind
  for warm-start priors and reports results back (#33).

## OrcaLab Deployment — Containerisation, Metrics, and Reference Docs

**PRs #34–37 · May 18, 2026**

### Added

- Expanded unit and performance test coverage with timeout-gap closure and
  benchmark tier for OrcaLab (#34).
- Deployed OrcaLab as a fully containerised service with Streamlit dashboard
  via Docker Compose (#35).
- Added real-time per-epoch metric streaming via a persistent metrics store
  (#36).
- Added API reference, deployment guide, and database schema reference to
  `docs/` (#37).

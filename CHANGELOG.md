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

## OrcaNet Foundations — Package Scaffold and Embedders

**PRs #38–41 · May 18–20, 2026**

### Added

- Scaffolded `orcanet` package for cross-domain knowledge transfer with
  module skeleton, config, and test infrastructure (#38).
- Implemented `CrossDomainEmbedder` using a DANN (Domain-Adversarial Neural
  Network) architecture for domain-invariant task representations (#39).
- Implemented `TextTaskEmbedder` for natural-language task-description
  embedding with statistical feature fusion (#40).
- Implemented GNN-based `ArchitectureEmbedder` for similarity-driven
  knowledge transfer across model architectures (#41).

## OrcaNet Transfer — CKA, Weight, Architecture, and Multi-Task Strategies

**PRs #42–45 · May 21–22, 2026**

### Added

- Implemented CKA (Centered Kernel Alignment) feature-transfer scoring and
  strategy infrastructure (#42).
- Implemented `WeightTransfer` strategy with layer-matching, scoring, deepcopy
  execution, and `get_optimizer_with_layer_lr` (#43).
- Implemented `ArchitectureTransfer` strategy with config adaptation and
  sequential model builder (#44).
- Implemented `MultiTaskTransfer` strategy with `MultiTaskModel` (shared
  backbone, task-specific heads), GradNorm weighting, and uncertainty
  weighting (#45).

### Fixed

- Guarded `add_task` against silent duplicate-task overwrite (#45).
- Detached feature tensors in `register_task_features` to prevent gradient
  leaks (#45).
- Corrected `execute_transfer` semantics and restored `nn.Module` return type
  in `WeightTransfer` (#43).

## OrcaNet Intelligence — Retrieval, Reasoning, API, and Integration

**PRs #46–51 · May 26–29, 2026**

### Added

- Implemented hybrid retrieval pipeline: `QueryExpander`, `LLMRanker`, and
  `HybridRetriever` three-stage pipeline (#46).
- Implemented LangChain-based reasoning agent with ReAct loop, four tool
  functions, prompt templates, response validators, and retry logic (#47).
- Built OrcaNet FastAPI service with eight HTTP endpoints (health, transfer,
  retrieve, explain, embed), dependency injection, middleware, and Pydantic
  response models (#48).
- Added `TransferPipeline` for three-way OrcaMind–OrcaLab–OrcaNet service
  coordination with `/api/v1/transfer/validate` endpoint (#49).
- Implemented `OrcaLabClient.create_experiment` and `wait_for_completion` in
  `orca-shared` (#49).
- Added Recall@10 retrieval benchmark, cross-domain embedding quality
  benchmark, and transfer recommendation quality benchmark (#50).
- Implemented cross-domain transfer demo notebook (#51).
- Added OrcaNet Dockerfile and deployment configuration (#51).

### Fixed

- Corrected `asyncio.gather` exception handling and expansion fan-out in
  retrieval pipeline (#46).
- Gated LLM health check behind `deep=true` query parameter (#48).
- Made lifespan shutdown cleanup fault-tolerant (#48).
- Offloaded `CrossDomainEmbedder.embed()` to thread pool to avoid blocking
  the event loop (#48).
- Replaced deprecated `get_event_loop()` with `get_running_loop()` in
  OrcaLab (#49).
- Upgraded Redis `depends_on` from `service_started` to `service_healthy`
  (#51).
- Seeded torch before embedder construction for deterministic benchmark
  weight initialisation (#50).

## Documentation Refinement — Technical Voice and Package READMEs

**PRs #52–54 · May 31 – Jun 7, 2026**

### Added

- Added per-package READMEs for `orca-shared`, `orcamind`, `orcalab`, and
  `orcanet`, plus a top-level `packages/` README (#53).
- Established the project's north-star aphorism in the root README (#54).

### Changed

- Sharpened technical voice and removed vague intensifiers across all eight
  `docs/` guides (#52).
- Enforced direct, consistent technical voice in all package READMEs (#53).
- Replaced absolute resilience claims with descriptions of implemented
  behaviour in package documentation (#53).
- Added `.claude/` to `.gitignore` (#52).

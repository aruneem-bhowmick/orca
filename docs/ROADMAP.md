# Roadmap

> Part of the [Orca](../README.md) meta-learning platform.

---

## OrcaMind — Next

- Dataset2Vec neural embedder (end-to-end from raw tabular data)
- Hydra config enhancements for distributed `orcamind train`

## OrcaLab — In Progress

**Done:**
- Package scaffold: full module skeleton, `pyproject.toml`, multi-stage Dockerfile, Typer CLI stub, Hydra config (`config.yaml`, `search/bayesian.yaml`, `pruner/asha.yaml`), and unit test suite
- Composable search space definitions: `Parameter` ABC and five concrete types (`IntParameter`, `FloatParameter`, `LogUniformParameter`, `DiscreteUniformParameter`, `CategoricalParameter`), `SearchSpace` with conditional parameter sampling and JSON persistence, `SearchSpaceComposer` with `merge`, `inherit`, and `restrict` — 44 unit tests, 100% line coverage on `search_spaces/`
- Search strategies: `SearchStrategy` ABC (`suggest` / `update` / `get_best` / `n_trials` contract, concrete `get_history()`); `RandomSearch` (seeded Optuna `RandomSampler`, FIFO pending-trial bookkeeping, `update()` validates param order); `GridSearch` (lazy Cartesian-product grid, per-type discretization rules, `StopIteration` on exhaustion, `TypeError` on unsupported parameter types); `BayesianSearch` (Optuna TPE sampler, `inject_priors()` warm-start via `optuna.trial.create_trial`, NaN/Inf-safe `update()`, direction-aware `get_best()`, SQLite/PostgreSQL persistence via `load_if_exists=True`, schema-stability guard on `suggest()`, finite-value guard in `inject_priors()`, `n >= 1` guard in `get_best()`); `EvolutionarySearch` (CMA-ES via `cma` library, encode/decode cycle mapping mixed parameter types to normalised `[0, 1]^d` vectors, one-hot categorical encoding, log-scale numeric support, population-lifecycle FIFO bookkeeping, convergence detection and restart from best-known point, validated `population_size`/`sigma0` inputs, single-space-instance contract, restart-while-pending guard, stale-queue eviction on restart); `MetaInformedSearch` (OrcaMind warm-start priors injected before first `suggest()`, fallback to Bayesian when no meta-priors available) — 78+ unit tests for search strategies
- Trial pruning strategies: `Pruner` ABC (`should_prune` / `name` contract, full `all_trial_values` cohort passed on every call); `MedianStoppingPruner` (configurable `warmup_steps`, peer-median comparison, best-value-up-to-step semantics, strict less-than boundary, self-exclusion from peer set); `ASHAPruner` (Li et al. 2018 — rung schedule at `min_resource × reduction_factor^k`, top-`1/reduction_factor` promotion, `_promoted` dict tracking, single-trial floor, >93% compute savings in sequential simulation, ≥40% savings asserted in test suite); `MetaPruner` (OrcaMind performance prediction as early-stopping signal, configurable `prediction_threshold` and `min_steps_before_prediction`, async/sync bridge via `asyncio.new_event_loop()`, graceful fallback to base pruner on any network error, exceptions never propagate from `should_prune`) — 90 unit tests
- Experiment lifecycle and execution: `ExperimentStatus` six-state enum (`PENDING → QUEUED → RUNNING → COMPLETED | FAILED | CANCELLED`); `ExperimentLifecycle` state machine with transition validation, UTC-timestamped audit log, and DB-first persistence ordering (in-memory state updated only after the repository write succeeds); `InvalidTransitionError` for forbidden edges; `Experiment` dataclass extending `ExperimentResult` with `arch_config`, `training_config`, and `tags`; `TrainableModel` protocol (`train_epoch(epoch) → float`); `ExperimentRunner` with injected `model_factory`, per-epoch MLflow metric streaming, `Pruner` integration with `"pruned"` reason on early stop, configurable `max_retries` (≥ 0) and `timeout` (> 0) with `ValueError` guards, retries transparent to the lifecycle (one `QUEUED → RUNNING` and one terminal transition regardless of attempt count); `BatchExperimentRunner` with `asyncio.Semaphore(max_parallel)` concurrency cap, input-order-preserving index-slot allocation, and failure isolation via returned `FAILED` results; `OrcaTracker.run_id` property (clears to `None` on `__aexit__` to prevent stale values) — 55 unit tests; **433+ unit tests total** across the package
- Prefect 2.x orchestration flows: four Prefect 2.x flows (`run_single_experiment`, `run_sweep`, `meta_informed_sweep`, `continuous_learning_loop`) and four tasks (`prepare_data`, `train_model`, `evaluate`, `log_results`) composing the runner, search strategies, pruners, storage, and OrcaMind into schedulable work-pool deployments; configurable strategy and pruner dispatch in sweep flows; OrcaMind warm-start and result flush in meta-informed sweep; continuous learning outer loop with per-iteration sleep; Prefect stub in `conftest.py` so all 50 orchestration unit tests run without a live Prefect install — 50 unit tests

**Next:**
- Live Streamlit dashboard with WebSocket metric streaming
- Bidirectional OrcaMind ↔ OrcaLab integration (priors in, results out)
- FastAPI service with 11 REST endpoints and WebSocket streaming

## OrcaNet — Planned

- Domain-adversarial cross-domain embedder (DANN)
- Transfer scoring via Centered Kernel Alignment (CKA)
- Hybrid retrieval: FAISS + PostgreSQL metadata filtering + LLM re-ranking
- LangChain reasoning agent for transfer explanations
- Three-way pipeline: OrcaNet → OrcaMind → OrcaLab

## Platform — Planned

- Kubernetes + Helm charts
- GitHub Actions CI/CD (lint, type-check, test, build, push image)
- Prometheus + Grafana monitoring

> Implementation details for current algorithms in [Components](COMPONENTS.md#core-algorithms).

---

## Reference Papers


| Algorithm   | Paper                                                                                   |
| ----------- | --------------------------------------------------------------------------------------- |
| MAML        | Finn et al., *Model-Agnostic Meta-Learning for Fast Adaptation of Deep Networks* (2017) |
| Reptile     | Nichol et al., *On First-Order Meta-Learning Algorithms* (2018)                         |
| Meta-SGD    | Li et al., *Meta-SGD: Learning to Learn Quickly for Few-Shot Learning* (2017)           |
| Dataset2Vec | Jomaa et al., *Dataset2Vec: Learning Dataset Meta-Features* (2021)                      |
| CKA         | Kornblith et al., *Similarity of Neural Network Representations Revisited* (2019)       |
| DANN        | Ganin et al., *Domain-Adversarial Training of Neural Networks* (2016)                   |
| ASHA        | Li et al., *A System for Massively Parallel Hyperparameter Tuning* (2018)               |
| CMA-ES      | Hansen, N., *The CMA Evolution Strategy: A Tutorial* (2016), arXiv:1604.00772           |

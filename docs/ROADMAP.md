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

**Next:**
- Search strategies: random, grid, Bayesian (Optuna TPE), evolutionary (CMA-ES), meta-informed (OrcaMind priors)
- ASHA pruning (target ≥40% compute reduction vs no pruning)
- Experiment lifecycle state machine and runner (MLflow tracking, retry logic, `BatchExperimentRunner`)
- Prefect 2.x orchestration flows: single experiment, sweep, meta-informed sweep, continuous learning
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

# Roadmap

> Part of the [Orca](../README.md) meta-learning platform.

---

## OrcaMind — Next

- Dataset2Vec neural embedder (end-to-end from raw tabular data)
- Hydra config enhancements for distributed `orcamind train`

## OrcaLab — Planned

- Adaptive hyperparameter search (Optuna with meta-priors from OrcaMind)
- Prefect 2.x orchestration flows: single experiment, sweep, meta-informed sweep, continuous learning
- ASHA pruning (target ≥40% compute reduction vs no pruning)
- Live Streamlit dashboard with WebSocket updates
- Bidirectional OrcaMind ↔ OrcaLab integration (priors in, results out)

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
| ASHA        | Li et al., *Massively Parallel Hyperparameter Tuning* (2018)                            |

# Orca

A meta-learning platform that treats prior experiments as training data.

---

Orca is a monorepo for meta-learning. The core premise: past experiments carry reusable signal, and a system that remembers them should outperform one that starts from scratch every time. Orca embeds ML tasks into a vector space, tracks what worked before, and uses that history to recommend models, warm-start training, and steer hyperparameter search.

The platform has three services and a shared infrastructure layer:


| Component       | Codename       | Role                                                                                                  |
| --------------- | -------------- | ----------------------------------------------------------------------------------------------------- |
| OrcaMind    | The Brain      | Meta-learning engine: task embedding, model selection, MAML/Reptile/Meta-SGD, warm-start transfer     |
| OrcaLab     | The Lab        | Experiment management: adaptive hyperparameter search, Prefect orchestration, live dashboards     |
| OrcaNet     | The Connector  | Cross-domain knowledge transfer: domain-invariant embeddings, LLM-powered reasoning, transfer scoring |
| orca-shared | The Foundation | Shared schemas, SQLAlchemy ORM, storage backends, MLflow wrappers, HTTP client library                |


---

## Quick Start

> For prerequisites and local dev setup, see [Getting Started](docs/GETTING-STARTED.md).

```bash
git clone https://github.com/AruneemB/orca.git
cd orca

# Start backing services
docker compose -f docker-compose.dev.yml up -d postgres redis minio mlflow
docker compose -f docker-compose.dev.yml run --rm orcamind python scripts/init_db.py
docker compose -f docker-compose.dev.yml up -d orcamind
```

Or with Make:

```bash
make install
make docker-up
```

---

## Documentation

| Guide | Description |
| ----- | ----------- |
| [Getting Started](docs/GETTING-STARTED.md) | Prerequisites, Docker Compose setup, local dev mode |
| [Components](docs/COMPONENTS.md) | orca-shared and OrcaMind internals, API, CLI, dashboard |
| [Architecture](docs/ARCHITECTURE.md) | System diagram, repo layout, tech stack |
| [Database](docs/DATABASE.md) | Alembic migrations, OpenML meta-dataset seeding |
| [Development](docs/DEVELOPMENT.md) | Testing, linting, type checking, pre-commit, Makefile |
| [Deployment](docs/DEPLOYMENT.md) | Environment variables, service topology, production notes |
| [API Reference](docs/API-REFERENCE.md) | REST endpoint specs for all three services |
| [Roadmap](docs/ROADMAP.md) | Planned features, reference papers |
| [Packages](packages/README.md) | Package-level READMEs for orca-shared, OrcaMind, OrcaLab, OrcaNet |

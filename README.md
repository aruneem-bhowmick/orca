# Orca

**A unified meta-learning platform. Teach machines how to learn, not just what to learn.**

---

Orca is a monorepo meta-learning ecosystem built around one idea: **prior experiments are a dataset, and we should learn from them**. Rather than starting every new ML task from scratch, Orca accumulates knowledge across tasks, embeds what it has seen, and uses that memory to recommend models, warm-start training, and guide hyperparameter search.

The ecosystem is composed of three interconnected services — **OrcaMind**, **OrcaLab**, and **OrcaNet** — plus a shared infrastructure layer used by all three.

---

## Components


| Component       | Codename       | Role                                                                                                  |
| --------------- | -------------- | ----------------------------------------------------------------------------------------------------- |
| **OrcaMind**    | The Brain      | Meta-learning engine: task embedding, model selection, MAML/Reptile/Meta-SGD, warm-start transfer     |
| **OrcaLab**     | The Lab        | Experiment management hub: adaptive hyperparameter search, Prefect orchestration, live dashboards     |
| **OrcaNet**     | The Connector  | Cross-domain knowledge transfer: domain-invariant embeddings, LLM-powered reasoning, transfer scoring |
| **orca-shared** | The Foundation | Shared schemas, SQLAlchemy ORM, storage backends, MLflow wrappers, HTTP client library                |


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
| [Roadmap](docs/ROADMAP.md) | Planned features, reference papers |

---

*Build the pod. Make it intelligent. Make it work together.*

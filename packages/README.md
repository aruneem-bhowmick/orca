# Packages

> Part of the [Orca](../README.md) meta-learning platform.

---

This directory contains the four Python packages that make up the Orca monorepo, managed as a [uv workspace](https://docs.astral.sh/uv/). Each package is independently installable, has its own test suite, and ships a multi-stage Dockerfile for containerised deployment.

## Package Map

```text
packages/
├── orca-shared/   Shared infrastructure — ORM, schemas, storage, tracking, HTTP clients
├── orcamind/      Meta-learning engine — task embedding, model selection, MAML/Reptile/Meta-SGD
├── orcalab/       Experiment orchestration — hyperparameter search, Prefect flows, live dashboards
└── orcanet/       Knowledge transfer agent — cross-domain embeddings, LLM reasoning, transfer scoring
```

## How the Packages Relate

```text
                 ┌───────────┐
                 │  OrcaNet  │  (port 8002)
                 └─────┬─────┘
                       │ retrieves source tasks, dispatches validation
          ┌────────────┼────────────┐
          ▼                         ▼
   ┌────────────┐           ┌────────────┐
   │  OrcaMind  │  ←─────→  │  OrcaLab   │
   │ (port 8000)│           │ (port 8001)│
   └──────┬─────┘           └──────┬─────┘
          │  priors / feedback      │
          └────────────┬────────────┘
                       ▼
              ┌────────────────┐
              │  orca-shared   │
              └────────────────┘
```

**orca-shared** is the foundation layer. It provides the PostgreSQL registry (SQLAlchemy ORM + async repository), Pydantic v2 schemas, MinIO/local storage backends, MLflow experiment tracking wrappers, and async HTTP clients used by the three services above it.

**OrcaMind** is the meta-learning engine. It embeds ML tasks into a vector space, trains meta-learners (MAML, Reptile, Meta-SGD), and recommends models for new tasks based on prior experiment performance.

**OrcaLab** is the experiment orchestration hub. It runs adaptive hyperparameter searches (Bayesian, evolutionary, grid, random, meta-informed), prunes unpromising trials early (ASHA, median, meta-pruner), and orchestrates training via Prefect flows. OrcaLab consumes model priors from OrcaMind and feeds trial results back, closing the meta-learning loop.

**OrcaNet** is the cross-domain knowledge transfer agent. It finds transferable source tasks using domain-adversarial embeddings and FAISS retrieval, scores transfer viability via CKA and weight-matching strategies, and explains recommendations through a LangChain ReAct agent. OrcaNet orchestrates both OrcaMind and OrcaLab for end-to-end transfer.

All inter-service communication degrades gracefully — each service continues to function independently when its dependencies are unreachable.

## Dependency Graph

```text
orcanet  →  orcalab  →  orcamind  →  orca-shared
                ↘          ↗
                 orca-shared
```

- `orca-shared` has no internal dependencies (only third-party libraries).
- `orcamind` depends on `orca-shared`.
- `orcalab` depends on `orca-shared` and `orcamind`.
- `orcanet` depends on all three.

## Installation

Install all packages from the workspace root:

```bash
make install
```

Or install individually:

```bash
uv pip install -e packages/orca-shared
uv pip install -e "packages/orcamind[dev]"
uv pip install -e "packages/orcalab[dev]"
uv pip install -e "packages/orcanet[dev]"
```

## Running Tests

```bash
# All packages
make test

# Individual packages
pytest packages/orca-shared/tests
pytest packages/orcamind/tests
pytest packages/orcalab/tests
pytest packages/orcanet/tests

# Unit tests only
make test-unit
```

See [Development](../docs/DEVELOPMENT.md) for the full testing, linting, and type-checking guide.

## Further Reading

| Document | Description |
|----------|-------------|
| [Architecture](../docs/ARCHITECTURE.md) | System diagram, repository structure, tech stack |
| [Components](../docs/COMPONENTS.md) | Implementation details for every module |
| [Getting Started](../docs/GETTING-STARTED.md) | Prerequisites, Docker Compose setup, local dev |
| [Deployment](../docs/DEPLOYMENT.md) | Environment variables, service topology, production notes |
| [Database](../docs/DATABASE.md) | Alembic migrations, schema reference, OpenML seeding |
| [API Reference](../docs/API-REFERENCE.md) | REST endpoint specifications for all three services |
| [Roadmap](../docs/ROADMAP.md) | Current progress and planned features |

---

For package-specific documentation, see the README in each subdirectory:

- [orca-shared](orca-shared/README.md)
- [orcamind](orcamind/README.md)
- [orcalab](orcalab/README.md)
- [orcanet](orcanet/README.md)

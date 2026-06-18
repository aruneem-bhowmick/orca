# scripts

> Operational scripts for the [Orca](../README.md) meta-learning platform.

---

This directory holds the one-off scripts you run to bring a fresh Orca deployment up to a working state: apply the database schema, create the Prefect work pool the sweep flows deploy onto, and seed the meta-learning registry with real benchmark tasks. None of the services import these files. You run them by hand or as Docker Compose pre-start steps.

## Layout

```text
scripts/
├── bootstrap_meta_dataset.py    # Seed the OrcaMind registry from OpenML CC-18 / CTR-23 and build the FAISS index
├── init_prefect.py              # Create the orcalab-pool Prefect work pool for sweep flow deployments
└── init_db.py                   # See note below; lives at packages/orcamind/scripts/init_db.py
```

### bootstrap_meta_dataset.py

Seeds the OrcaMind registry with real benchmark tasks so the recommendation and similarity endpoints return data instead of empty results. For every task it downloads, it extracts features from the raw dataset, computes a 25-dimensional statistical embedding, runs five baseline models with five-fold cross-validation, and persists `Task`, `Model`, `Experiment`, and `Performance` rows through the repository layer. Each embedding goes into an in-memory FAISS cosine-similarity index, which is written to disk at the end of the run.

The five baselines are logistic regression, random forest, XGBoost, an RBF SVM, and KNN. The SVM and SVR variants are skipped for datasets above 10,000 samples, where their training cost stops being worth it. Individual task-download and per-model failures are logged and skipped, so one unreachable task cannot abort the whole run.

```bash
python scripts/bootstrap_meta_dataset.py \
  --suites cc18 ctr23 \
  --max-tasks 20 \
  --output-dir data/
```

| Flag | Default | Description |
|------|---------|-------------|
| `--suites` | `cc18 ctr23` | OpenML benchmark suites to download |
| `--max-tasks INT` | all | Cap on tasks per suite |
| `--output-dir PATH` | `data/` | Directory for the FAISS index output |
| `--db-url URL` | from `ORCA_DB_URL` | Override the PostgreSQL connection URL |
| `--dry-run` | off | Download, embed, and score without writing to the registry |

`--dry-run` parses and scores every task but skips the registry writes and the index save. Use it to confirm a suite downloads cleanly before committing the results. The connection URL falls back to a local dev default when neither `--db-url` nor `ORCA_DB_URL` is set.

The `openml` dependency is imported lazily inside the functions that need it, so importing the module (and running its test suite) does not require `openml` to be installed. Seeding details and the schema it writes are documented in [Database](../docs/DATABASE.md#bootstrap-the-meta-dataset).

### init_prefect.py

Creates the `orcalab-pool` work pool (type `process`) that the OrcaLab sweep deployments target. It is a one-time setup step per Prefect server instance: run it once after the Prefect server is up and before you start a worker.

```bash
python scripts/init_prefect.py
```

The call is a thin wrapper around `prefect work-pool create orcalab-pool --type process` with `check=True`, so a non-zero exit from the `prefect` CLI propagates as a `CalledProcessError`. Worker setup is covered in [Deployment](../docs/DEPLOYMENT.md#prefect-worker-setup).

### init_db.py

This script is not in this directory. It lives at `packages/orcamind/scripts/init_db.py` because it resolves `alembic.ini` relative to its own path inside the OrcaMind package. It is referenced throughout the docs and the Docker Compose setup as `python scripts/init_db.py` because the orcamind container sets its working directory to the package root, where the path resolves. It runs `alembic upgrade head` against the database URL in `DATABASE_URL` and exits non-zero on any failure, which makes it safe to call as a Compose pre-start step. Migration details are in [Database](../docs/DATABASE.md).

## Usage

The intended order, against a running Docker Compose stack, is:

```bash
# 1. Apply database migrations (run inside the orcamind container)
docker compose -f docker-compose.dev.yml run --rm orcamind python scripts/init_db.py

# 2. Start OrcaMind, then create the Prefect work pool
docker compose -f docker-compose.dev.yml up -d orcamind
python scripts/init_prefect.py

# 3. Seed the meta-dataset (small subset first, then the full suites)
python scripts/bootstrap_meta_dataset.py --suites cc18 --max-tasks 20 --output-dir data/
```

The full bring-up sequence, including health checks and which services depend on which, is in [Getting Started](../docs/GETTING-STARTED.md).

## Testing

```bash
python -m pytest scripts/tests --cov=scripts --cov-report=term-missing
```

The suite covers argument parsing, the OpenML download and baseline-evaluation helpers, the registry-storage coroutines, and the async orchestration loop. `openml` is mocked throughout, so the tests run without network access or the `openml` package installed. See [Development](../docs/DEVELOPMENT.md) for how these tests fit into the wider suite.

## Further Reading

| Document | Description |
|----------|-------------|
| [Getting Started](../docs/GETTING-STARTED.md) | Prerequisites, Docker Compose setup, local dev mode |
| [Database](../docs/DATABASE.md) | Alembic migrations, schema reference, OpenML seeding |
| [Deployment](../docs/DEPLOYMENT.md) | Environment variables, service topology, Prefect worker setup |
| [Architecture](../docs/ARCHITECTURE.md) | System diagram, repository layout, tech stack |
| [Development](../docs/DEVELOPMENT.md) | Testing, linting, type checking, pre-commit hooks |
| [OrcaMind](../packages/orcamind/README.md) | The meta-learning engine these scripts seed and serve |

---

[Back to root](../README.md)

# OrcaMind

> The meta-learning engine of the [Orca](../../README.md) platform. Codename: **The Brain**.

---

OrcaMind learns from prior experiments to recommend models and training configurations for new ML tasks. It embeds tasks into a vector space, measures similarity, and applies meta-learning algorithms (MAML, Reptile, Meta-SGD) to accelerate convergence on unseen problems — turning the history of past experiments into a reusable knowledge base.

OrcaMind serves as the intelligence backbone for the other Orca services: [OrcaLab](../orcalab/README.md) queries it for model priors to warm-start hyperparameter searches, and [OrcaNet](../orcanet/README.md) uses it to retrieve source tasks and predict transfer performance.

## Architecture

```text
orcamind/
├── core/           Meta-learning algorithms (MAML, Reptile, Meta-SGD, Warm-start)
├── embedders/      Task embedding strategies (statistical, neural, FAISS similarity)
├── selectors/      Model selection (nearest-neighbour, performance prediction, ranking)
├── training/       Meta-training loop (MetaTrainer, task sampling, callbacks, metrics)
├── api/            FastAPI REST service — 13 endpoints across 6 routers (port 8000)
├── dashboard/      Streamlit analytics app — 4 pages (port 8501)
├── cli.py          Typer CLI — init, serve, dashboard, train, recommend
├── alembic/        Database migrations for the shared registry
├── scripts/        init_db.py (runs alembic upgrade head)
└── config/         Hydra YAML configs (model, dataset, optimizer)
```

### Core Algorithms

Four meta-learning implementations, all extending a common `MetaLearningAlgorithm` ABC:

| Algorithm | Description | Reference |
|-----------|-------------|-----------|
| **MAML** | Model-Agnostic Meta-Learning with second-order gradients | Finn et al. 2017 |
| **Reptile** | First-order approximation via averaged SGD trajectories | Nichol et al. 2018 |
| **Meta-SGD** | Per-parameter learnable learning rates | Li et al. 2017 |
| **Warm-start Transfer** | Direct weight initialisation from similar tasks | — |

### Task Embedders

- **StatisticalEmbedder** — hand-crafted 25-dimensional feature vector from dataset statistics (sample count, feature count, class balance, etc.).
- **NeuralEmbedder** — learned embeddings via a PyTorch encoder trained on the meta-dataset.
- **FaissIndex** — approximate nearest-neighbour search over embedding vectors for fast similarity retrieval.

### Model Selectors

- **NearestNeighborSelector** — k-NN lookup in the task embedding space to find similar tasks and their best models.
- **PerformancePredictor** — regression model predicting expected accuracy from task and model features.
- **Ranker** — ranking strategy combining multiple scoring signals.

### Training

`MetaTrainer` orchestrates the meta-training loop with configurable task sampling, early stopping and checkpoint callbacks, and metric computation. Training configuration is managed via Hydra YAML files.

## API

OrcaMind exposes a FastAPI REST service on port 8000 with 13 endpoints:

| Router | Key Endpoints |
|--------|--------------|
| Tasks | Register tasks, retrieve by ID/domain, get task embeddings |
| Models | Query model registry |
| Recommend | Model recommendation for a given task |
| Feedback | Receive trial results from OrcaLab |
| Performances | Performance metric queries |
| Adapt | Model adaptation triggers |

Interactive API docs are available at `http://localhost:8000/docs`.

See [API Reference](../../docs/API-REFERENCE.md) for the full endpoint specification.

## Dashboard

A Streamlit analytics application (port 8501) with four pages:

- **Task Browser** — explore registered tasks and their embeddings.
- **Recommendation Explorer** — query and visualise model recommendations.
- **Performance Heatmap** — model-vs-task performance matrix.
- **Training Progress** — live meta-training metrics.

## CLI

```bash
orcamind init            # Initialise workspace directories
orcamind serve           # Start the FastAPI server (--reload for dev)
orcamind dashboard       # Launch the Streamlit dashboard
orcamind train           # Run the meta-training loop
orcamind recommend       # Get model recommendations for a dataset
```

## Database Migrations

OrcaMind owns the Alembic migration environment for the shared PostgreSQL registry. All seven tables used by the Orca ecosystem are defined here.

```bash
# Apply migrations
cd packages/orcamind && alembic upgrade head

# Or via Docker
docker compose -f docker-compose.dev.yml run --rm orcamind python scripts/init_db.py
```

See [Database](../../docs/DATABASE.md) for the full schema reference, revision history, and OpenML seeding instructions.

## Configuration

Hydra YAML configs under `config/`:

```text
config/
├── config.yaml          Root config (paths, mlflow, seed, device)
├── model/maml.yaml      MAML algorithm parameters
├── dataset/openml.yaml  OpenML dataset configuration
└── optimizer/adam.yaml   Adam optimizer settings
```

## Integration Points

| Direction | Mechanism |
|-----------|-----------|
| **OrcaMind → OrcaLab** | Model recommendations warm-start Bayesian search via `MetaInformedSearch` |
| **OrcaLab → OrcaMind** | Trial results feed back via `POST /api/v1/feedback` |
| **OrcaNet → OrcaMind** | Source task retrieval and model queries for transfer scoring |

Both directions are resilient — OrcaLab starts sweeps without priors when OrcaMind is unreachable, and OrcaNet returns transfer recommendations even without OrcaMind data.

See [Architecture](../../docs/ARCHITECTURE.md) for the full integration diagram.

## Testing

```bash
pytest packages/orcamind/tests/unit         # Unit tests (40+ files)
pytest packages/orcamind/tests/integration  # API and Docker smoke tests
```

See [Development](../../docs/DEVELOPMENT.md) for the full testing guide.

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Meta-learning | PyTorch, PyTorch Lightning, learn2learn, higher |
| ML | scikit-learn, XGBoost, SciPy |
| Vector search | FAISS |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Config | Hydra, OmegaConf |
| Tracking | MLflow |
| Dashboard | Streamlit, Plotly |
| CLI | Typer, Rich |
| Data | OpenML |
| Shared | [orca-shared](../orca-shared/README.md) |

---

[Back to packages](../README.md)

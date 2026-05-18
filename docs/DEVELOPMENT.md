# Development

> Part of the [Orca](../README.md) meta-learning platform.

---

## Running Tests

```bash
# All tests (all packages)
pytest packages/ -v --cov

# OrcaMind unit tests (no services required)
pytest packages/orcamind/tests/unit/ -v

# OrcaLab unit tests (no services required)
pytest packages/orcalab/tests/unit/ -v

# OrcaLab experiment lifecycle and runner tests only
pytest packages/orcalab/tests/unit/experiments/ -v

# OrcaLab pruning unit tests only
pytest packages/orcalab/tests/unit/pruning/ -v

# OrcaLab orchestration tests only (no Prefect install required)
pytest packages/orcalab/tests/unit/orchestration/ -v

# OrcaLab visualization tests only (no Streamlit or Plotly install required)
pytest packages/orcalab/tests/unit/visualization/ -v

# OrcaLab REST API integration tests (no services required — all deps mocked)
pytest packages/orcalab/tests/integration/api/ -v

# OrcaMind ↔ OrcaLab bidirectional integration tests (no services required — OrcaMind HTTP mocked with respx)
pytest packages/orcalab/tests/integration/ -v

# OrcaMind integration tests (requires docker-compose stack)
pytest packages/orcamind/tests/integration/ -v

# OrcaLab performance / benchmark tests (no services required)
pytest packages/orcalab/tests/performance/ -v

# OrcaLab deployment validation tests (no services required — reads config files and Python module)
pytest packages/orcalab/tests/unit/test_dockerfile.py \
       packages/orcalab/tests/unit/test_docker_compose.py \
       packages/orcalab/tests/unit/test_init_prefect.py \
       packages/orcalab/tests/unit/test_app_module_export.py -v

# OrcaNet unit tests (no services required)
pytest packages/orcanet/tests/unit/ -v

# OrcaNet — package structure and importability
pytest packages/orcanet/tests/unit/test_package.py -v

# OrcaNet — CLI smoke tests
pytest packages/orcanet/tests/unit/test_cli.py -v

# OrcaNet — Hydra config validation
pytest packages/orcanet/tests/unit/test_config.py -v
```

The test suite has 80+ test files across unit, integration, performance, and deployment-validation categories.

The OrcaLab API integration tests run without a live database, Prefect server, or MLflow instance. An `ASGITransport` client fixture pre-populates `app.state` manually (bypassing the ASGI lifespan) and overrides all dependency providers via `dependency_overrides`, so tests exercise the full request/response cycle including middleware, routing, and validation while every external call goes to an `AsyncMock`.

The OrcaMind ↔ OrcaLab bidirectional integration tests run without a live OrcaMind service. `respx` intercepts all `httpx` calls and routes them to pre-configured mock responses, so the full client/task/strategy call chain is exercised at the network layer without running any external process. A lightweight Prefect stub in `tests/integration/conftest.py` replaces the `prefect` module in `sys.modules`, supporting both `@task(...)` and bare `@task` decorator forms.

The visualization unit tests run without a live Streamlit or Plotly install. A session-scoped `_patch_streamlit` fixture in `tests/unit/visualization/conftest.py` replaces both libraries in `sys.modules` before any page or component module is imported, so the pure data-processing functions can be tested independently of the Streamlit runtime.

OrcaMind integration tests auto-skip when their target service port is unreachable — run `make docker-up` first to exercise them.

**OrcaNet test patterns:**

- *Parametrized submodule imports* — `test_package.py` collapses six structurally identical submodule import assertions into a single `@pytest.mark.parametrize("submodule", [...])` test. Adding a new submodule requires only a new entry in the parameter list.
- *Per-test CLI runner fixture* — `test_cli.py` exposes the Typer `CliRunner` as a pytest fixture (`scope="function"`) rather than a module-level variable, preventing any runner state from leaking between tests.
- *pyproject.toml anchor path resolution* — `test_config.py` locates the `config/` directory by walking ancestor directories until a `pyproject.toml` is found, rather than using a hard-coded `parents[N]` depth index. This remains correct regardless of where the test file moves within the repository tree.

The performance benchmark tests in `tests/performance/` make executable compute-efficiency assertions that cannot be expressed as ordinary unit tests. They drive deterministic synthetic sweeps — no external services, no randomness — and enforce measurable invariants about algorithm behaviour at scale. Currently the tier contains `TestASHAPruningSavings`, which simulates 20-trial hyperparameter sweeps on a concave-quadratic learning-curve objective and asserts that ASHA executes ≤60% of the steps an unpruned baseline would require (≥40% compute savings). The scaling test additionally runs a 27-trial cohort and asserts that savings for the larger cohort are at least as good as for the 20-trial baseline, enforcing the monotonicity property directly.

---

## Docker Deployment

### Starting the Full Stack

```bash
# 1. Start backing services
docker compose -f docker-compose.dev.yml up -d postgres redis minio mlflow prefect

# 2. Wait for each service to pass its healthcheck, then start OrcaMind
docker compose -f docker-compose.dev.yml up -d orcamind

# 3. Create the Prefect work pool that sweep flows are deployed onto
python scripts/init_prefect.py

# 4. Start OrcaLab API (waits for all dependencies including OrcaMind to be healthy)
docker compose -f docker-compose.dev.yml up -d orcalab

# 5. Start the Streamlit dashboard (waits for OrcaLab API to be healthy)
docker compose -f docker-compose.dev.yml up -d orcalab-dashboard

# 6. Start OrcaNet (waits for postgres, orcamind, and orcalab to be healthy)
docker compose -f docker-compose.dev.yml up -d orcanet
```

### Verifying the Deployment

```bash
# OrcaMind
curl http://localhost:8000/health
# → {"status":"healthy","db":true,"faiss":false,"mlflow":true}

# OrcaLab API
curl http://localhost:8001/health
# → {"status":"healthy","db":true,"prefect":true}

# OrcaNet
curl http://localhost:8002/health
# → {"status":"ok","orcamind":"http://orcamind:8000","orcalab":"http://orcalab:8001"}

# Run a test sweep (use_orcamind: false skips the OrcaMind warm-start)
curl -X POST http://localhost:8001/api/v1/sweeps \
  -H 'Content-Type: application/json' \
  -d '{"task_id": "test-task-1", "n_trials": 5, "use_orcamind": false}'
# → {"sweep_id": "<uuid>"}

# Poll status
curl http://localhost:8001/api/v1/sweeps/<sweep_id>
# → {"sweep_id":"...","n_trials_total":5,"n_completed":5,...}

# Fetch results
curl http://localhost:8001/api/v1/sweeps/<sweep_id>/results
# → [{"trial_id":"...","objective":...,"params":{...}}, ...]

# OrcaLab Dashboard — open in browser
# http://localhost:8502
```

### Deployment Test Suite

The deployment validation tests run without Docker — they inspect config files and the Python module graph:

```bash
# Dockerfile structure
pytest packages/orcalab/tests/unit/test_dockerfile.py -v

# docker-compose.dev.yml service configuration
pytest packages/orcalab/tests/unit/test_docker_compose.py -v

# Prefect work-pool initialisation script
pytest packages/orcalab/tests/unit/test_init_prefect.py -v

# Module-level app export (uvicorn entrypoint)
pytest packages/orcalab/tests/unit/test_app_module_export.py -v

# All deployment tests together
pytest packages/orcalab/tests/unit/test_dockerfile.py \
       packages/orcalab/tests/unit/test_docker_compose.py \
       packages/orcalab/tests/unit/test_init_prefect.py \
       packages/orcalab/tests/unit/test_app_module_export.py -v
```

---

## Linting and Type Checking

```bash
ruff check .          # Lint
ruff format .         # Format
mypy packages/        # Type check (strict on orca-shared)
```

---

## Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install    # Install on git commit + push hooks
pre-commit run --all-files
```

Hooks run: ruff lint, ruff format, mypy. The push stage runs the unit test suite.

---

## Makefile Targets

```bash
make install      # uv venv + install all packages
make test         # pytest with coverage
make lint         # ruff check
make type-check   # mypy
make docker-up    # docker compose up -d
make docker-down  # docker compose down
make docker-logs  # docker compose logs -f
make clean        # remove __pycache__, .pytest_cache, .mypy_cache
make help         # list all targets
```

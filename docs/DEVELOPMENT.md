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
```

The test suite has 80+ test files across unit and integration categories.

The OrcaLab API integration tests run without a live database, Prefect server, or MLflow instance. An `ASGITransport` client fixture pre-populates `app.state` manually (bypassing the ASGI lifespan) and overrides all dependency providers via `dependency_overrides`, so tests exercise the full request/response cycle including middleware, routing, and validation while every external call goes to an `AsyncMock`.

The OrcaMind ↔ OrcaLab bidirectional integration tests run without a live OrcaMind service. `respx` intercepts all `httpx` calls and routes them to pre-configured mock responses, so the full client/task/strategy call chain is exercised at the network layer without running any external process. A lightweight Prefect stub in `tests/integration/conftest.py` replaces the `prefect` module in `sys.modules`, supporting both `@task(...)` and bare `@task` decorator forms.

The visualization unit tests run without a live Streamlit or Plotly install. A session-scoped `_patch_streamlit` fixture in `tests/unit/visualization/conftest.py` replaces both libraries in `sys.modules` before any page or component module is imported, so the pure data-processing functions can be tested independently of the Streamlit runtime.

OrcaMind integration tests auto-skip when their target service port is unreachable — run `make docker-up` first to exercise them.

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

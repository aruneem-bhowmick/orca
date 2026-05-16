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

# OrcaLab pruning unit tests only
pytest packages/orcalab/tests/unit/pruning/ -v

# Integration tests (requires docker-compose stack)
pytest packages/orcamind/tests/integration/ -v
```

The test suite has 59+ test files across unit and integration categories. Integration tests auto-skip when their target service port is unreachable — run `make docker-up` first to exercise them.

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

.PHONY: install test test-unit test-integration lint format type-check \
        pre-commit-install docker-up docker-down docker-logs clean help

UV := uv

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-22s\033[0m %s\n", $$1, $$2}'

install: ## Install all workspace packages in editable mode
	$(UV) venv
	$(UV) pip install -e packages/orca-shared
	$(UV) pip install -e "packages/orcamind[dev]"

test: ## Run all tests with coverage
	$(UV) run pytest

test-unit: ## Run unit tests only
	$(UV) run pytest packages/orcamind/tests/unit packages/orca-shared/tests -x -q

test-integration: ## Run integration tests (requires docker-up)
	$(UV) run pytest packages/orcamind/tests/integration -x -q

lint: ## Run ruff linter
	$(UV) run ruff check .

format: ## Auto-format code with ruff
	$(UV) run ruff format .

type-check: ## Run mypy type checker
	$(UV) run mypy packages/orca-shared/

pre-commit-install: ## Install pre-commit hooks
	$(UV) run pre-commit install

docker-up: ## Start development services (postgres, redis, minio, mlflow, prefect)
	docker compose -f docker-compose.dev.yml up -d

docker-down: ## Stop development services
	docker compose -f docker-compose.dev.yml down

docker-logs: ## Tail logs from all dev services
	docker compose -f docker-compose.dev.yml logs -f

clean: ## Remove __pycache__, .pyc, and coverage artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name ".pytest_cache" -delete
	find . -type d -name "*.egg-info" -delete
	rm -f .coverage
	rm -rf htmlcov/

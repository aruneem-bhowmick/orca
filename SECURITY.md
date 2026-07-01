# Security

Orca's security model, credential handling, trust boundaries, and operational guidance for the full ecosystem.

## Reporting vulnerabilities

If you find a security issue, do not open a public GitHub issue. Use GitHub's private vulnerability reporting instead: navigate to the **Security** tab of this repository and select **Report a vulnerability**. Include a description of the issue, reproduction steps, and the affected component or service. We will acknowledge receipt within 48 hours and provide a fix timeline within 7 days.

## Architecture overview

Orca is a multi-service ecosystem. Each component runs as an independent process, communicates over HTTP, and is containerized separately.

| Service | Default port | Role |
|---|---|---|
| **Orca UI** | 5173 | React SPA — user-facing frontend |
| **Orca Web (BFF)** | 8003 | Backend-for-frontend — the only authenticated, public-facing API gateway |
| **OrcaMind** | 8000 | Meta-learning engine — internal service |
| **OrcaLab** | 8001 | Experiment management — internal service |
| **OrcaNet** | 8002 | Cross-domain transfer + LLM reasoning — internal service |

Infrastructure services used by the ecosystem:

| Service | Purpose |
|---|---|
| PostgreSQL 15 | Shared metadata registry; user accounts and sessions |
| Redis 7 | Cache and task queues |
| MinIO | S3-compatible artifact storage (model checkpoints, datasets) |
| MLflow | Experiment tracking (metadata in PostgreSQL, artifacts in MinIO) |
| Prefect | Workflow orchestration for OrcaLab experiment flows |

### Trust boundary

The **Orca Web BFF** is the only service that should receive traffic from outside the private deployment network. All user authentication happens here. OrcaMind, OrcaLab, OrcaNet, PostgreSQL, Redis, MinIO, MLflow, and Prefect must sit behind a private network boundary. Exposing any of them directly to the public internet is a misconfiguration.

In the Docker Compose stack, all services share a single bridge network (`orca-dev-network`). In Kubernetes, network policies must enforce equivalent isolation.

## Authentication

### User authentication (BFF)

Orca Web authenticates users with JSON Web Tokens using the HS256 algorithm. The JWT secret is loaded from the `JWT_SECRET_KEY` environment variable at startup.

**Access tokens** are short-lived (15 minutes by default) and are returned in the response body after a successful login or token refresh. Clients include them as `Authorization: Bearer <token>` on every protected request. The `get_current_user` dependency in `orca_web/api/deps.py` validates the token signature, confirms the `type` claim is `"access"`, and verifies the user still exists and is active.

**Refresh tokens** are long-lived (7 days by default) and are delivered as an `httponly` cookie scoped to the `/api/v1/auth` path. A per-token JTI (JWT ID) is stored in the `user_sessions` PostgreSQL table. On each call to `POST /auth/refresh`, the old JTI is revoked and a new refresh token is issued — stolen refresh tokens cannot be silently reused. `SessionRepository.revoke_all_for_user()` invalidates every active session for a user, and should be called on password change or suspected compromise.

The `secure` flag on the refresh cookie is `False` in the default configuration and must be set to `True` in any deployment that terminates TLS. See the hardening checklist at the end of this document.

### OAuth 2.0 (Google and GitHub)

Orca Web supports OAuth 2.0 login via Google and GitHub, implemented with `authlib`. OAuth providers are registered only when the corresponding client ID environment variable is non-empty, so unconfigured providers are unavailable at runtime. After the provider callback, the BFF upserts the user record (matching by OAuth `sub` claim, then by email) and issues the same JWT pair as for local login. OAuth users have no `password_hash`.

### Local password authentication

Passwords are hashed with bcrypt via `passlib`. Plaintext passwords are never stored, logged, or compared directly. The `verify_password` function performs a constant-time comparison.

### Role model

Users have a `role` column with two valid values: `"user"` (default) and `"admin"`. Profile access is enforced by `GET /users/{user_id}`, which rejects requests where the caller is neither the target user nor an admin. Role-gating for other endpoints follows the same pattern via the `get_current_user` dependency.

### WebSocket authentication

The live experiment metric stream at `/orcalab/ws/experiments/{id}/live` accepts the JWT access token either as a `?token=<jwt>` query parameter or in the `Sec-WebSocket-Protocol` header (for environments where query parameters are not available). Invalid or missing tokens receive close code 4001.

### Internal service authentication

OrcaMind, OrcaLab, and OrcaNet expose unauthenticated HTTP APIs. They trust all inbound requests. This is a known gap; API key authentication for inter-service calls is planned. Until it is implemented, **network isolation is the only defense layer for these services**. No request that originates outside the private deployment network should reach them.

## Credential handling

All secrets are sourced from environment variables. No secret appears in any configuration file that is (or could be) committed to source control. Dev-only defaults in `config.py` and `docker-compose.dev.yml` are clearly marked as placeholders and must be replaced before any production deployment.

| Secret | Environment variable | Used by |
|---|---|---|
| JWT signing key | `JWT_SECRET_KEY` | Orca Web |
| Database connection | `DATABASE_URL` | All services |
| Redis connection | `REDIS_URL` | Orca Web, OrcaMind |
| MinIO access key | `MINIO_ACCESS_KEY` | OrcaMind |
| MinIO secret key | `MINIO_SECRET_KEY` | OrcaMind |
| MLflow tracking URI | `MLFLOW_TRACKING_URI` | OrcaMind, OrcaLab |
| OpenAI API key | `OPENAI_API_KEY` | OrcaNet |
| Anthropic API key | `ANTHROPIC_API_KEY` | OrcaNet |
| Google OAuth client ID | `GOOGLE_CLIENT_ID` | Orca Web |
| Google OAuth client secret | `GOOGLE_CLIENT_SECRET` | Orca Web |
| GitHub OAuth client ID | `GITHUB_CLIENT_ID` | Orca Web |
| GitHub OAuth client secret | `GITHUB_CLIENT_SECRET` | Orca Web |

**What is never logged:** JWT tokens, refresh tokens, passwords, OAuth secrets, database connection strings, MinIO credentials, LLM API keys.

**What is logged:** HTTP method, path, status code, and elapsed time (request logging middleware); upstream service health check results; WebSocket relay start and end events with experiment ID and user ID (no token values); aggregator errors with the failing URL and exception type.

## Data handling

### What Orca stores

**PostgreSQL (shared registry):** task definitions and metadata, task embeddings, model configurations, experiment records, performance metrics, transfer mappings, and search space definitions. None of these tables store raw dataset content; `dataset_uri` holds a pointer to the artifact in MinIO.

**PostgreSQL (Orca Web):** user accounts (email, bcrypt-hashed password, OAuth provider/sub), active refresh token sessions (JTI, expiry, revoked flag, optional IP address and device info), activity log entries, and user bookmarks.

**MinIO:** ML model artifacts (checkpoints, weights), dataset files, and MLflow run artifacts. Access requires valid MinIO credentials. Bucket permissions are the deployer's responsibility.

**Redis:** Prefect task queues, transient cache. No long-lived sensitive data is designed to persist here beyond queue lifetime.

### What Orca sends to external APIs

OrcaNet sends task descriptions, domain labels, task types, and similarity context to a configured LLM provider for transfer reasoning and retrieval reranking. The active provider is a user-selectable runtime option; OpenAI and Anthropic are currently supported, with additional providers planned.

No user personal information (email addresses, usernames, passwords, OAuth tokens) is included in LLM prompts. The data sent consists of ML task metadata derived from the shared registry.

### What Orca does not store

Raw dataset content is not stored in PostgreSQL. Model checkpoints and dataset files in MinIO are keyed by URI, not replicated into the registry. No PR content, diff, or user-authored code from external repositories is processed by Orca.

## Network and CORS

CORS is configured in `orca_web/api/middleware.py` from the `CORS_ORIGINS` environment variable (comma-separated list of allowed origins). When origins are specified, `allow_credentials=True` is set and the list is used exactly as provided. When `CORS_ORIGINS` is empty or unset, the middleware falls back to `allow_origins=["*"]` with `allow_credentials=False`. The wildcard fallback is suitable only for local development; production deployments must set `CORS_ORIGINS` to the exact frontend origin.

The BFF exposes `GET /health` without authentication. It checks connectivity to PostgreSQL, Redis, OrcaMind, OrcaLab, and OrcaNet and returns a degraded status if any backing service is unreachable. This endpoint leaks service topology information and should not be exposed to the public internet in production.

## Container security

All Orca service Dockerfiles use a two-stage build pattern:

1. A `builder` stage installs Python dependencies using `uv` into an isolated virtual environment at `/opt/venv`.
2. A `runtime` stage starts from `python:3.11-slim`, copies only `/opt/venv` and application source from the builder, and creates a dedicated non-root user (`orca`, uid 1001, gid 1001) to run the process.

No secrets are baked into any image. All credentials are injected at runtime via environment variables. Files copied into the runtime image are owned by the `orca` user. The `USER orca` directive ensures the process never runs as root.

Each container declares a health check that probes its own `/health` endpoint. Docker Compose and Kubernetes use these probes to manage service startup order and readiness.

## Code quality gates

These checks run on every pull request and must pass before merging:

- **Ruff**: linter and formatter (enforces style, import ordering, and common error patterns)
- **Mypy**: type checking in strict mode for `orca-shared`; standard mode for other packages
- **pytest**: minimum 80% coverage threshold per package (`fail_under = 80`)

Pre-commit hooks run Ruff and Mypy on every commit. Unit tests run as a pre-push hook across `packages/orcamind/tests/unit` and `packages/orca-shared/tests`.

## Supply chain

Runtime dependencies are declared in each package's `pyproject.toml` with minimum version constraints. Development dependencies (testing, linting, type-checking) are declared under `[project.optional-dependencies] dev` and are not installed in production images.

| Dependency | Purpose | Used by |
|---|---|---|
| `fastapi` / `uvicorn` | HTTP server framework | All services |
| `pydantic` / `pydantic-settings` | Schema validation and settings | All services |
| `sqlalchemy` / `asyncpg` | Async database access | All services |
| `alembic` | Database schema migrations | Orca Web, OrcaMind, OrcaLab |
| `httpx` | Async HTTP client (inter-service calls) | Orca Web, OrcaNet |
| `python-jose[cryptography]` | JWT encoding and decoding | Orca Web |
| `passlib[bcrypt]` | Password hashing | Orca Web |
| `authlib` | OAuth 2.0 client | Orca Web |
| `redis` | Redis client | Orca Web, OrcaMind |
| `websockets` | WebSocket relay (upstream OrcaLab) | Orca Web |
| `torch` / `pytorch-lightning` | Deep learning framework | OrcaMind, OrcaNet |
| `faiss-cpu` | Vector similarity search | OrcaMind, OrcaNet |
| `sentence-transformers` | Text embeddings | OrcaNet |
| `langchain` / `langchain-openai` / `langchain-anthropic` | LLM agent framework | OrcaNet |
| `prefect` | Workflow orchestration | OrcaLab |
| `optuna` | Hyperparameter search | OrcaLab |
| `mlflow` | Experiment tracking | OrcaMind, OrcaLab |
| `minio` | MinIO / S3 client | OrcaMind |
| `hydra-core` / `omegaconf` | Hierarchical configuration | OrcaMind, OrcaLab, OrcaNet |

## Threat model

### Trusted inputs

- Environment variables set by the deployer
- Docker Compose or Kubernetes configuration files
- Database contents (assuming the private network boundary is intact)
- Configuration files in the repository (Hydra configs, `pyproject.toml`)

### Untrusted inputs

**HTTP request bodies and parameters (BFF):** All user-supplied input arrives at the BFF and is validated by Pydantic schemas before any processing occurs. Invalid input is rejected with a structured error response. SQLAlchemy's parameterized queries prevent SQL injection; raw string interpolation into queries does not occur.

**OAuth callback payloads:** Provider responses are validated by authlib using the provider's published OIDC metadata or explicit token URL. The callback only exchanges a short-lived authorization code, never a reused credential.

**LLM API responses:** OrcaNet parses LLM provider responses through Pydantic validators. Responses are treated as structured data, never executed. Unexpected fields or type mismatches raise validation errors.

**Upstream service responses (internal):** Because internal services are unauthenticated (see above), a compromised internal service could return malicious responses to the BFF. The BFF parses upstream JSON and forwards it to the browser; it does not execute it.

### Out of scope

**LLM provider data retention.** Task metadata sent to OpenAI or Anthropic is subject to those providers' terms of service and data processing agreements. Orca does not control how providers store, process, or retain that data. Review the relevant provider's policies independently before sending sensitive task information.

**Secrets in task datasets.** If a dataset registered with Orca contains secrets or personally identifiable information, those values may be included in task metadata forwarded to the configured LLM provider. Orca does not scan or redact dataset content before sending it to external APIs.

**MinIO bucket exposure.** Orca does not configure MinIO bucket policies. A misconfigured MinIO instance with public bucket access would expose all stored artifacts. Bucket policy management is the deployer's responsibility.

**Internal service compromise.** If an attacker reaches OrcaMind, OrcaLab, or OrcaNet through a network boundary failure, those services have no additional authentication layer. Containment depends entirely on network isolation being intact.

**Denial of service via large inputs.** Orca enforces no global limit on task dataset size, experiment batch size, or LLM prompt length in the current implementation. Resource consumption bounds are the deployer's responsibility at the infrastructure level.

## Deployment hardening checklist

For any deployment where the BFF is reachable from outside a trusted private network:

- [ ] Set `JWT_SECRET_KEY` to a cryptographically random value of at least 32 bytes; never use the default `dev-secret-change-in-prod`
- [ ] Terminate TLS at a reverse proxy (nginx, Caddy, ALB, or equivalent) in front of the BFF; set `secure=True` on the refresh token cookie (`orca_web/api/routers/auth.py`, `_set_refresh_cookie`)
- [ ] Set `CORS_ORIGINS` to the exact production frontend URL; do not rely on the wildcard fallback
- [ ] Place OrcaMind, OrcaLab, OrcaNet, PostgreSQL, Redis, MinIO, MLflow, and Prefect behind a private network boundary with no inbound access from the public internet
- [ ] Replace all dev-mode credential placeholders: `POSTGRES_PASSWORD`, `MINIO_ROOT_USER`, `MINIO_ROOT_PASSWORD`, and any service-level `MINIO_ACCESS_KEY` / `MINIO_SECRET_KEY` values
- [ ] Store LLM API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) and OAuth secrets in a secrets manager or your deployment platform's secrets store — not in Compose files or `.env` files committed to source control
- [ ] Enable PostgreSQL TLS and use connection strings with `sslmode=require`
- [ ] Set a Redis `AUTH` password if Redis is reachable from any network segment outside the deployment host
- [ ] Configure MinIO bucket policies to deny public access
- [ ] Pin Docker base images to specific digests for reproducible, auditable builds
- [ ] Rotate `JWT_SECRET_KEY` periodically — all active user sessions are invalidated on rotation
- [ ] Monitor application logs for sustained 401/403 response rates, upstream health check failures, and WebSocket relay errors
- [ ] Review the data handling policy of your configured LLM provider before registering tasks that contain sensitive or proprietary information

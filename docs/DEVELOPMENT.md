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

# OrcaNet — CrossDomainEmbedder, GRL, TextTaskEmbedder, and ArchitectureEmbedder unit tests (no services required)
pytest packages/orcanet/tests/unit/embeddings/ -v

# OrcaNet — TextTaskEmbedder only
pytest packages/orcanet/tests/unit/embeddings/test_text_features.py -v

# OrcaNet — ArchitectureGraph and ArchitectureEmbedder unit tests only
pytest packages/orcanet/tests/unit/embeddings/test_architecture_embedder.py -v

# OrcaNet — transfer module unit tests only (all four strategies: FeatureTransfer, WeightTransfer, ArchitectureTransfer, MultiTaskTransfer)
pytest packages/orcanet/tests/unit/transfer/ -v

# OrcaNet — linear_cka correctness tests only
pytest packages/orcanet/tests/unit/transfer/test_feature_transfer.py -v -k "TestLinearCKA"

# OrcaNet — FeatureTransfer scoring, guards, metadata, and execute_transfer tests only
pytest packages/orcanet/tests/unit/transfer/test_feature_transfer.py -v -k "TestFeatureTransfer or TestTransferScore or TestExecuteTransfer"

# OrcaNet — WeightTransfer tests only (all classes)
pytest packages/orcanet/tests/unit/transfer/test_weight_transfer.py -v

# OrcaNet — WeightTransfer scoring tests only (identical architecture + match_by modes + structure)
pytest packages/orcanet/tests/unit/transfer/test_weight_transfer.py -v -k "TestWeightTransferScore"

# OrcaNet — WeightTransfer execute_transfer and optimizer tests only
pytest packages/orcanet/tests/unit/transfer/test_weight_transfer.py -v -k "TestWeightTransferExecute or TestGetOptimizer"

# OrcaNet — WeightTransfer guards and metadata tests only
pytest packages/orcanet/tests/unit/transfer/test_weight_transfer.py -v -k "TestWeightTransferGuards or TestWeightTransferMetadata"

# OrcaNet — ArchitectureTransfer tests only (all classes)
pytest packages/orcanet/tests/unit/transfer/test_architecture_transfer.py -v

# OrcaNet — ArchitectureTransfer adapt_architecture tests only
pytest packages/orcanet/tests/unit/transfer/test_architecture_transfer.py -v -k "TestAdaptArchitecture"

# OrcaNet — ArchitectureTransfer score_transfer tests only
pytest packages/orcanet/tests/unit/transfer/test_architecture_transfer.py -v -k "TestArchitectureTransferScore"

# OrcaNet — ArchitectureTransfer execute_transfer, metadata, and guards tests only
pytest packages/orcanet/tests/unit/transfer/test_architecture_transfer.py -v -k "TestArchitectureTransferExecute or TestArchitectureTransferMetadata or TestArchitectureTransferGuards"

# OrcaNet — MultiTaskTransfer and MultiTaskModel tests only (all classes)
pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v

# OrcaNet — MultiTaskModel forward routing and loss tests only
pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v -k "TestMultiTaskModelForward or TestMultiTaskModelLoss"

# OrcaNet — MultiTaskModel uncertainty loss and log_sigma gradient-flow tests only
pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v -k "TestMultiTaskModelUncertaintyLoss"

# OrcaNet — _get_backbone_out_dim helper tests only
pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v -k "TestGetBackboneOutDim"

# OrcaNet — MultiTaskTransfer add_task, score_transfer, execute_transfer, and metadata tests only
pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v -k "TestAddTask or TestScoreTransfer or TestExecuteTransfer or TestTransferMetadata"

# OrcaNet — GradNorm weight renormalisation tests only
pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v -k "TestGradnormWeighting"

# OrcaNet — Uncertainty weighting convergence tests (gradient direction + 10-step mini-training loop)
pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v -k "TestUncertaintyWeighting"
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
- *Training-mode preservation testing* — `test_cross_domain.py` explicitly tests that `CrossDomainEmbedder.embed()` does not permanently mutate the model's training state. Two complementary assertions cover both directions: a model in `.train()` mode returns to training mode after `embed()` finishes, and a model already in `.eval()` stays in eval mode. This matters because `embed()` is commonly called from inside a training loop for online evaluation, and a silent mode flip would corrupt subsequent gradient computation.
- *Domain invariance as a geometric assertion* — `TestDomainInvariance.test_within_vs_cross_domain_spread` quantifies domain invariance by comparing the standard deviation of cosine distances within each domain against the standard deviation of cross-domain distances, asserting a ratio in [0.3, 3.0]. This avoids the brittleness of testing exact cluster assignments while still enforcing the geometric property — the absence of domain clustering — that the OrcaNet retrieval layer relies on.
- *Offline SentenceTransformer stub* — `tests/unit/embeddings/conftest.py` patches `orcanet.embeddings.text_features.SentenceTransformer` with `_DeterministicSentenceTransformer` via a session-scoped autouse fixture. The stub never downloads model weights: it maps a 36-keyword domain vocabulary (vision, financial, medical, NLP, general ML) to fixed vector dimensions and fills the remaining 348 dimensions with low-amplitude deterministic noise (σ = 0.05, seeded by `hash(text)`). Semantic ordering is preserved by construction — image-domain keywords are orthogonal to financial-domain keywords — so similarity-ordering tests hold without a network connection or a local model cache.
- *Relative similarity assertion over fixed thresholds* — `test_architecture_embedder.py` tests cross-architecture separation by asserting `embedder.similarity(a, a) > embedder.similarity(a, b)` rather than `similarity < 0.9`. Hardcoded cosine thresholds are fragile across random weight seeds and optional message-passing backends (dense adjacency vs. `GCNConv`); the relative ordering holds for any valid embedder, making the test robust without weakening the invariant it enforces.
- *Boundary tests for public API contracts* — the architecture embedder test suite covers `top_k=0` (must return `[]`) and `top_k=-1` (must raise `ValueError`) explicitly. Without the `ValueError` guard, Python's negative-slice semantics would silently return a near-full result list. The boundary tests pin this contract so future refactors cannot regress it.
- *Relaxed CKA threshold for shallow networks* — `TestFeatureTransferRandomModels.test_overall_below_identical` asserts `score.overall < 0.8` rather than `< 0.5`. Two independently initialised shallow MLPs share no learned structure, but the shared input distribution still induces a consistent covariance, producing CKA ≈ 0.60. The 0.8 threshold captures the meaningful gap from identical models (≈1.0) without over-specifying the exact value, which shifts with network depth and width.
- *execute_transfer source-immutability guard* — `test_does_not_mutate_source_model` snapshots all source parameters before the call and asserts byte-level equality afterwards. This catches weight-patching bugs that corrupt the source model in-place even when the returned adapted model looks correct.
- *Geometric CKA orthogonality assertion* — `TestLinearCKAOrthogonal._orthogonal_pair` builds the test pair from non-overlapping columns of a QR-factored random matrix, guaranteeing exact orthogonality by construction rather than relying on probabilistic near-orthogonality. The resulting CKA < 0.1 assertion is a precise algebraic claim, not a statistical threshold.
- *Deepcopy-semantics split in WeightTransfer tests* — `test_weight_transfer.py` recognises that `execute_transfer` starts from `deepcopy(source_model)`, so shape mismatches can never occur within that path and the no-raise guarantee is trivially satisfied for the deepcopy case. Rather than building a misleading mismatched-architecture fixture, the test file separates concerns: `test_shape_mismatch_skipped_without_exception` verifies that no exception is raised across all three `match_by` modes using a matched architecture (the real goal), while `test_safe_reinit_handles_1d_and_2d_tensors` targets `_safe_reinit` directly with 1-D and 2-D tensors to pin the reinitialisation path. Making the separation explicit prevents future readers from wondering why the deepcopy fixture "doesn't really test mismatches."
- *Per-parameter optimizer group verification* — `TestGetOptimizerWithLayerLR.test_transferred_params_get_decayed_lr` resolves each optimizer `param_groups` entry back to its parameter name via `model.named_parameters()` and asserts the learning rate individually. This catches silent index-drift bugs where a list comprehension over `named_parameters()` assigns the wrong LR to the wrong parameter — a class of bug that would pass a coarser "check the mean LR" assertion but silently mis-train transferred layers.
- *match_by mode isolation* — `TestWeightTransferScoreMatchBy` constructs a single pair of models with differing `out_dim` (first layer identical, last layer shape-mismatched) and asserts each `match_by` mode independently. `"name"` mode gives `overall == 1.0` (names all exist regardless of shape), `"both"` drops below 1.0 (last-layer params excluded) while preserving 1.0 for the first-layer params, and `"shape"` produces a valid float without a specific assertion (shape-first matching on arbitrary architectures is deterministic but architecturally coupled). Testing each mode against the same fixture makes it straightforward to verify that adding a new mode does not silently change the behaviour of the existing three.
- *AsyncMock + synchronous ABC bridge testing* — `test_architecture_transfer.py` uses `AsyncMock` for `OrcaMindClient` (whose `get_best_model` is a coroutine) alongside `MagicMock(spec=ArchitectureEmbedder)` for the synchronous embedder. The `_run_coro` helper that bridges the synchronous `TransferStrategy` ABC with the async client is exercised automatically by every `score_transfer` call in the test suite, verifying that it completes without deadlock even though `AsyncMock` resolves immediately.
- *Activation-function coverage* — `test_all_activation_types_supported` builds a model from `_config_with_activations()`, which sequences all four supported activation types (`sigmoid`, `tanh`, `gelu`, `none`) in a single forward pass. A concrete output-shape assertion (`(2, 5)`) verifies that `_build_sequential_from_config` threads `current_in` correctly through activation modules (which do not change the width) without off-by-one errors in the layer dimension chain.
- *`nn.ModuleDict` parameter-registration assertion* — `TestMultiTaskModelForward.test_head_params_in_model_parameters` iterates every head parameter and asserts its `id()` appears in the set of `id(p) for p in model.parameters()`. This pins the requirement that task heads are stored as `nn.ModuleDict` (not a plain `dict`), because a plain dict would silently exclude head parameters from the optimiser, producing a model that trains only the backbone.
- *`nn.ParameterDict` gradient-flow assertion* — `TestMultiTaskModelUncertaintyLoss.test_log_sigmas_have_grad_after_backward` calls `compute_uncertainty_loss(batch).backward()` and asserts that `log_sigma.grad is not None` for each task. This pins the requirement that log-sigma scalars are stored as `nn.ParameterDict` (not plain tensors), because a plain tensor would have `requires_grad=False` by default and would never receive a gradient.
- *Uncertainty convergence test using a mini-training loop* — `TestUncertaintyWeighting.test_noisy_task_learns_higher_log_sigma` runs 10 Adam steps with a fixed random seed. One task always sees constant labels (low cross-entropy), the other sees random labels (irreducible noise, high cross-entropy). After training, it asserts `log_sigma_hard > log_sigma_easy`. The test uses small dimensions (4-dim input, 8 hidden, 2 classes) so it completes in milliseconds. A fixed seed makes the assertion deterministic. This pattern is preferred over testing the gradient direction alone because it verifies the *integrated effect* of the Kendall et al. 2018 objective over multiple steps, not just a single-step gradient sign.
- *`register_task_features` as a pre-registration pattern* — `TestScoreTransfer` tests both the fallback path (no features registered → neutral 0.5 score) and the live path (features registered → real cosine similarity computed via `CrossDomainEmbedder.embed()`). Testing the fallback explicitly pins the contract that `score_transfer` never raises when called without prior setup, mirroring the same policy enforced in `FeatureTransfer` (raises on missing probe data) and `WeightTransfer` (raises on missing model) — where `MultiTaskTransfer` takes the more lenient approach of returning a neutral score.
- *Backbone output-dim inference boundary test* — `TestGetBackboneOutDim.test_raises_when_no_linear` passes a bare `nn.ReLU()` to `_get_backbone_out_dim` and asserts a `ValueError`. This ensures that a backbone without any `nn.Linear` (e.g. a pre-activation residual block that exposes only `nn.Conv2d`) produces an actionable error at construction time rather than a silent failure when `add_task` is first called.

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

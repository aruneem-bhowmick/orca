# Development

> Part of the [Orca](../README.md) meta-learning platform.

---

## Running Tests

```bash
# All tests (all packages + scripts)
uv run pytest -v --cov

# OrcaMind unit tests (no services required)
uv run pytest packages/orcamind/tests/unit/ -v

# OrcaLab unit tests (no services required)
uv run pytest packages/orcalab/tests/unit/ -v

# OrcaLab experiment lifecycle and runner tests only
uv run pytest packages/orcalab/tests/unit/experiments/ -v

# OrcaLab pruning unit tests only
uv run pytest packages/orcalab/tests/unit/pruning/ -v

# OrcaLab orchestration tests only (no Prefect install required)
uv run pytest packages/orcalab/tests/unit/orchestration/ -v

# OrcaLab visualization tests only (no Streamlit or Plotly install required)
uv run pytest packages/orcalab/tests/unit/visualization/ -v

# OrcaLab REST API integration tests (no services required — all deps mocked)
uv run pytest packages/orcalab/tests/integration/api/ -v

# OrcaMind ↔ OrcaLab bidirectional integration tests (no services required — OrcaMind HTTP mocked with respx)
uv run pytest packages/orcalab/tests/integration/ -v

# OrcaMind integration tests (requires docker-compose stack)
uv run pytest packages/orcamind/tests/integration/ -v

# OrcaLab performance / benchmark tests (no services required)
uv run pytest packages/orcalab/tests/performance/ -v

# OrcaLab deployment validation tests (no services required — reads config files and Python module)
uv run pytest packages/orcalab/tests/unit/test_dockerfile.py \
       packages/orcalab/tests/unit/test_docker_compose.py \
       packages/orcalab/tests/unit/test_init_prefect.py \
       packages/orcalab/tests/unit/test_app_module_export.py -v

# OrcaNet unit tests (no services required)
uv run pytest packages/orcanet/tests/unit/ -v

# OrcaNet — package structure and importability
uv run pytest packages/orcanet/tests/unit/test_package.py -v

# OrcaNet — CLI smoke tests
uv run pytest packages/orcanet/tests/unit/test_cli.py -v

# OrcaNet — Hydra config validation
uv run pytest packages/orcanet/tests/unit/test_config.py -v

# OrcaNet — deployment configuration validation (docker-compose.dev.yml structure, Dockerfile,
#            env var names in main.py; no services required)
uv run pytest packages/orcanet/tests/unit/test_deployment_config.py -v

# OrcaNet — demo notebook structure and content validation (no services required)
uv run pytest packages/orcanet/tests/unit/test_notebook.py -v

# OrcaNet — CrossDomainEmbedder, GRL, TextTaskEmbedder, and ArchitectureEmbedder unit tests (no services required)
uv run pytest packages/orcanet/tests/unit/embeddings/ -v

# OrcaNet — TextTaskEmbedder only
uv run pytest packages/orcanet/tests/unit/embeddings/test_text_features.py -v

# OrcaNet — ArchitectureGraph and ArchitectureEmbedder unit tests only
uv run pytest packages/orcanet/tests/unit/embeddings/test_architecture_embedder.py -v

# OrcaNet — transfer module unit tests only (all four strategies: FeatureTransfer, WeightTransfer, ArchitectureTransfer, MultiTaskTransfer)
uv run pytest packages/orcanet/tests/unit/transfer/ -v

# OrcaNet — linear_cka correctness tests only
uv run pytest packages/orcanet/tests/unit/transfer/test_feature_transfer.py -v -k "TestLinearCKA"

# OrcaNet — FeatureTransfer scoring, guards, metadata, and execute_transfer tests only
uv run pytest packages/orcanet/tests/unit/transfer/test_feature_transfer.py -v -k "TestFeatureTransfer or TestTransferScore or TestExecuteTransfer"

# OrcaNet — WeightTransfer tests only (all classes)
uv run pytest packages/orcanet/tests/unit/transfer/test_weight_transfer.py -v

# OrcaNet — WeightTransfer scoring tests only (identical architecture + match_by modes + structure)
uv run pytest packages/orcanet/tests/unit/transfer/test_weight_transfer.py -v -k "TestWeightTransferScore"

# OrcaNet — WeightTransfer execute_transfer and optimizer tests only
uv run pytest packages/orcanet/tests/unit/transfer/test_weight_transfer.py -v -k "TestWeightTransferExecute or TestGetOptimizer"

# OrcaNet — WeightTransfer guards and metadata tests only
uv run pytest packages/orcanet/tests/unit/transfer/test_weight_transfer.py -v -k "TestWeightTransferGuards or TestWeightTransferMetadata"

# OrcaNet — ArchitectureTransfer tests only (all classes)
uv run pytest packages/orcanet/tests/unit/transfer/test_architecture_transfer.py -v

# OrcaNet — ArchitectureTransfer adapt_architecture tests only
uv run pytest packages/orcanet/tests/unit/transfer/test_architecture_transfer.py -v -k "TestAdaptArchitecture"

# OrcaNet — ArchitectureTransfer score_transfer tests only
uv run pytest packages/orcanet/tests/unit/transfer/test_architecture_transfer.py -v -k "TestArchitectureTransferScore"

# OrcaNet — ArchitectureTransfer execute_transfer, metadata, and guards tests only
uv run pytest packages/orcanet/tests/unit/transfer/test_architecture_transfer.py -v -k "TestArchitectureTransferExecute or TestArchitectureTransferMetadata or TestArchitectureTransferGuards"

# OrcaNet — MultiTaskTransfer and MultiTaskModel tests only (all classes)
uv run pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v

# OrcaNet — MultiTaskModel forward routing and loss tests only
uv run pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v -k "TestMultiTaskModelForward or TestMultiTaskModelLoss"

# OrcaNet — MultiTaskModel uncertainty loss and log_sigma gradient-flow tests only
uv run pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v -k "TestMultiTaskModelUncertaintyLoss"

# OrcaNet — _get_backbone_out_dim helper tests only
uv run pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v -k "TestGetBackboneOutDim"

# OrcaNet — MultiTaskTransfer add_task, score_transfer, execute_transfer, and metadata tests only
uv run pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v -k "TestAddTask or TestScoreTransfer or TestExecuteTransfer or TestTransferMetadata"

# OrcaNet — GradNorm weight renormalisation tests only
uv run pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v -k "TestGradnormWeighting"

# OrcaNet — Uncertainty weighting convergence tests (gradient direction + 10-step mini-training loop)
uv run pytest packages/orcanet/tests/unit/transfer/test_multi_task.py -v -k "TestUncertaintyWeighting"

# OrcaNet — retrieval module unit tests only (all three files: QueryExpander, LLMRanker, HybridRetriever)
uv run pytest packages/orcanet/tests/unit/retrieval/ -v

# OrcaNet — QueryExpander and _parse_list_from_response tests only
uv run pytest packages/orcanet/tests/unit/retrieval/test_query_expander.py -v

# OrcaNet — LLMRanker and _parse_ranked_list tests only
uv run pytest packages/orcanet/tests/unit/retrieval/test_ranker.py -v

# OrcaNet — HybridRetriever, _task_to_feature_vector, and _deduplicate_and_sort tests only
uv run pytest packages/orcanet/tests/unit/retrieval/test_retriever.py -v

# OrcaNet — _parse_list_from_response helper tests only (list format stripping)
uv run pytest packages/orcanet/tests/unit/retrieval/test_query_expander.py -v -k "TestParseListFromResponse"

# OrcaNet — QueryExpander LLM invocation and prompt format tests only
uv run pytest packages/orcanet/tests/unit/retrieval/test_query_expander.py -v -k "TestQueryExpander"

# OrcaNet — _parse_ranked_list JSON parsing and validation tests only
uv run pytest packages/orcanet/tests/unit/retrieval/test_ranker.py -v -k "TestParseRankedList"

# OrcaNet — LLMRanker.rerank() sorting, top_k, prompt, and short-circuit tests only
uv run pytest packages/orcanet/tests/unit/retrieval/test_ranker.py -v -k "TestLLMRanker"

# OrcaNet — _task_to_feature_vector helper tests only (shape, encoding, None handling)
uv run pytest packages/orcanet/tests/unit/retrieval/test_retriever.py -v -k "TestTaskToFeatureVector"

# OrcaNet — _deduplicate_and_sort helper tests only
uv run pytest packages/orcanet/tests/unit/retrieval/test_retriever.py -v -k "TestDeduplicateAndSort"

# OrcaNet — HybridRetriever.retrieve() and retrieve_with_expanded_queries() tests only
uv run pytest packages/orcanet/tests/unit/retrieval/test_retriever.py -v -k "TestHybridRetriever"

# OrcaNet — all three benchmark modules (no services required)
uv run pytest packages/orcanet/tests/benchmarks/ -v

# OrcaNet — Recall@9 retrieval benchmark only
uv run pytest packages/orcanet/tests/benchmarks/test_retrieval_benchmark.py -v

# OrcaNet — cross-domain embedding quality benchmark only
uv run pytest packages/orcanet/tests/benchmarks/test_embedding_benchmark.py -v

# OrcaNet — transfer recommendation quality benchmark only
uv run pytest packages/orcanet/tests/benchmarks/test_transfer_quality.py -v

# OrcaNet — full test suite with coverage enforcement (unit + benchmarks)
uv run pytest packages/orcanet/ --cov=orcanet --cov-fail-under=80

# OrcaNet — skip benchmark tests (fast unit-only run)
uv run pytest packages/orcanet/tests/unit/ -v -m "not benchmark"

# Orca Web — all tests (no services required — all deps mocked)
uv run pytest packages/orca-web/tests/ -v

# Orca Web — app factory and health endpoint tests only
uv run pytest packages/orca-web/tests/test_main.py -v

# Orca Web — auth endpoint tests only
uv run pytest packages/orca-web/tests/test_auth.py -v

# Orca Web — dashboard endpoint tests only
uv run pytest packages/orca-web/tests/test_dashboard.py -v

# Orca Web — user endpoint tests only
uv run pytest packages/orca-web/tests/test_users.py -v

# Orca Web — proxy utility tests only (shared forwarding + activity logging)
uv run pytest packages/orca-web/tests/test_proxy_utils.py -v

# Orca Web — OrcaMind proxy endpoint tests only
uv run pytest packages/orca-web/tests/test_proxy_orcamind.py -v

# Orca Web — OrcaLab proxy endpoint tests only
uv run pytest packages/orca-web/tests/test_proxy_orcalab.py -v

# Orca Web — OrcaNet proxy endpoint tests only
uv run pytest packages/orca-web/tests/test_proxy_orcanet.py -v

# Orca Web — all proxy tests together
uv run pytest packages/orca-web/tests/test_proxy_utils.py \
       packages/orca-web/tests/test_proxy_orcamind.py \
       packages/orca-web/tests/test_proxy_orcalab.py \
       packages/orca-web/tests/test_proxy_orcanet.py -v

# Orca Web — full suite with coverage enforcement (≥80%)
uv run pytest packages/orca-web/tests/ -v --cov=orca_web --cov-fail-under=80

# Scripts — bootstrap_meta_dataset.py + init_prefect.py (55 tests, ≥80% coverage)
uv run pytest scripts/tests/ -v --cov=scripts --cov-report=term-missing
```

The test suite spans 80+ test files across unit, integration, performance, deployment-validation, proxy, and scripts categories.

The OrcaLab API integration tests run without a live database, Prefect server, or MLflow instance. An `ASGITransport` client fixture pre-populates `app.state` manually (bypassing the ASGI lifespan) and overrides all dependency providers via `dependency_overrides`, so tests exercise the full request/response cycle including middleware, routing, and validation while every external call goes to an `AsyncMock`.

The OrcaMind ↔ OrcaLab bidirectional integration tests run without a live OrcaMind service. `respx` intercepts all `httpx` calls and routes them to pre-configured mock responses, so the full client/task/strategy call chain is exercised at the network layer without running any external process. A lightweight Prefect stub in `tests/integration/conftest.py` replaces the `prefect` module in `sys.modules`, supporting both `@task(...)` and bare `@task` decorator forms.

The visualization unit tests run without a live Streamlit or Plotly install. A session-scoped `_patch_streamlit` fixture in `tests/unit/visualization/conftest.py` replaces both libraries in `sys.modules` before any page or component module is imported, so the pure data-processing functions can be tested independently of the Streamlit runtime.

OrcaMind integration tests auto-skip when their target service port is unreachable — run `make docker-up` first to exercise them.

**OrcaNet test patterns:**

- *Parametrized submodule imports* — `test_package.py` collapses six structurally identical submodule import assertions into a single `@pytest.mark.parametrize("submodule", [...])` test. Adding a new submodule requires only a new entry in the parameter list.
- *Per-test CLI runner fixture* — `test_cli.py` exposes the Typer `CliRunner` as a pytest fixture (`scope="function"`) rather than a module-level variable, preventing any runner state from leaking between tests.
- *pyproject.toml anchor path resolution* — `test_config.py` locates the `config/` directory by walking ancestor directories until a `pyproject.toml` is found, rather than using a hard-coded `parents[N]` depth index. This remains correct regardless of where the test file moves within the repository tree.
- *docker-compose.dev.yml root path resolution* — `test_deployment_config.py` locates the compose file by walking ancestor directories until `docker-compose.dev.yml` is found. The same anchor-walk pattern is used for the Dockerfile, making all deployment config tests relocatable within the repo tree.
- *Source-code inspection for env var names* — `TestEnvVarReading` in `test_deployment_config.py` uses `inspect.getsource` to assert that `main.py` contains `"ORCAMIND_API_URL"` and `"ORCALAB_API_URL"` and does NOT contain the legacy `"ORCAMIND_URL"` / `"ORCALAB_URL"` strings. This catches regressions where someone reverts the env-var alignment without a test failure.
- *Notebook JSON content assertions* — `test_notebook.py` parses the demo notebook as JSON and asserts structural properties (≥ 8 code cells, section headers 1–8, valid `nbformat` 4) plus content invariants: no `# TODO: implement` stubs remain, all eight API endpoints are referenced, matplotlib is used, the UMAP import is guarded with `try/except ImportError`, and a demo fallback is present for offline use. No cells are executed during tests.
- *Training-mode preservation testing* — `test_cross_domain.py` explicitly tests that `CrossDomainEmbedder.embed()` does not permanently mutate the model's training state. Two complementary assertions cover both directions: a model in `.train()` mode returns to training mode after `embed()` finishes, and a model already in `.eval()` stays in eval mode. This matters because `embed()` is commonly called from inside a training loop for online evaluation, and a silent mode flip would corrupt subsequent gradient computation.
- *Domain invariance as a geometric assertion* — `TestDomainInvariance.test_within_vs_cross_domain_spread` quantifies domain invariance by comparing the standard deviation of cosine distances within each domain against the standard deviation of cross-domain distances, asserting a ratio in [0.3, 3.0]. This avoids the brittleness of testing exact cluster assignments while still enforcing the geometric property — the absence of domain clustering — that the OrcaNet retrieval layer relies on.
- *Offline SentenceTransformer stub* — `tests/unit/embeddings/conftest.py` patches `orcanet.embeddings.text_features.SentenceTransformer` with `_DeterministicSentenceTransformer` via a session-scoped autouse fixture. The stub never downloads model weights: it maps a 36-keyword domain vocabulary (vision, financial, medical, NLP, general ML) to fixed vector dimensions and fills the remaining 348 dimensions with low-amplitude deterministic noise (σ = 0.05, seeded by `hash(text)`). Semantic ordering is preserved by construction — image-domain keywords are orthogonal to financial-domain keywords — so similarity-ordering tests hold without a network connection or a local model cache.
- *Relative similarity assertion over fixed thresholds* — `test_architecture_embedder.py` tests cross-architecture separation by asserting `embedder.similarity(a, a) > embedder.similarity(a, b)` rather than `similarity < 0.9`. Hardcoded cosine thresholds are fragile across random weight seeds and optional message-passing backends (dense adjacency vs. `GCNConv`); the relative ordering holds for any valid embedder, keeping the invariant in place without a hardcoded threshold.
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
- *Backbone output-dim inference boundary test* — `TestGetBackboneOutDim.test_raises_when_no_linear` passes a bare `nn.ReLU()` to `_get_backbone_out_dim` and asserts a `ValueError`. This ensures that a backbone without any `nn.Linear` (e.g. a pre-activation residual block that exposes only `nn.Conv2d`) raises an error at construction time rather than failing silently when `add_task` is first called.
- *Duplicate task-registration guard* — `TestAddTask.test_raises_on_duplicate_task_id` calls `add_task` twice for the same task and asserts a `ValueError` on the second call. Without the guard, the second call would silently replace any trained head parameters and reset the log-sigma to zero for uncertainty weighting; the test pins the fail-fast contract so the bug surfaces at call time rather than manifesting as a mysterious loss spike after reload.
- *Shared-backbone gradient accumulation* — `TestMultiTaskModelForward.test_backbone_shared_across_heads` runs a forward pass through both task heads, sums the outputs, calls `backward()`, and asserts that at least one backbone parameter has a non-None gradient. This replaces the previous tautological `model.backbone is model.backbone` check and directly verifies that both head paths contribute gradients to the same backbone module — a requirement that would be silently violated if `execute_transfer` accidentally copied the backbone instead of sharing it.
- *GradNorm partial-update isolation* — `TestGradnormWeighting.test_partial_update_preserves_omitted_task_weight` registers three tasks, then calls `update_gradnorm_weights` with only two of them. It asserts that the third task's weight is bit-identical after the call. This pins the invariant that a partial gradient-norm update is safe to call within a training step where only a subset of tasks produced usable gradient signals, without silently perturbing the weights of tasks that were not included. A companion test (`test_raises_on_unknown_task_id`) asserts that supplying an unregistered task id raises `ValueError` rather than silently inserting a spurious key.
- *`SimpleNamespace` mock factory for multi-dependency injection* — `test_retriever.py` defines `_make_mocks(task)` which returns all five `HybridRetriever` dependencies as a `SimpleNamespace`. Each test calls `_make_mocks` and `_build_retriever` to get a ready retriever with sensible defaults, then mutates only the attributes it cares about (e.g. `mocks.repo.get_by_id = AsyncMock(side_effect=...)`). This avoids deeply nested fixture pyramids while keeping each test's setup visible in its own body — a useful pattern whenever a class under test has more than two or three injected collaborators.
- *Real tensor in embedder mock* — `HybridRetriever.retrieve` chains `.squeeze(0).detach().numpy()` on the value returned by `CrossDomainEmbedder.embed`. The embedder mock uses `embedder.embed.return_value = torch.zeros(25)` (a real 1-D tensor) rather than a `MagicMock` chain: `squeeze(0)` on a 1-D tensor is a no-op, `detach()` returns the same tensor, and `numpy()` produces a valid array. This gives the FAISS index mock a proper numpy array without needing a complex spec-restricted mock for `CrossDomainEmbedder`.
- *`side_effect` lambda for multi-task repository mocking* — when a test needs `TaskRepository.get_by_id` to return different tasks for different UUIDs (e.g. the similarity-threshold test), `repo.get_by_id = AsyncMock(side_effect=lambda uid: task_map.get(uid))` routes calls through a pre-populated dict keyed by `UUID` objects. This is preferable to `side_effect=[task1, task2]` (call-order dependent) and to separate mock instances per call site (not possible with a single `AsyncMock`).
- *Pydantic field-constraint boundary test* — `TestParseRankedList.test_score_out_of_range_returns_empty_list` submits `"score": 1.5` to exercise `_RankedItem`'s `Field(ge=0.0, le=1.0)` constraint. The resulting `ValidationError` is caught by `_parse_ranked_list` and silently converted to `[]`. Testing the boundary explicitly pins the graceful-degradation contract: a misbehaving LLM that emits an out-of-range score must never propagate a `ValidationError` to the caller, and without the boundary test a future refactor could remove or weaken the `Field` constraint without any test catching the regression.
- *No-LLM-call assertion for empty candidate list* — `TestLLMRanker.test_empty_candidates_returns_empty_without_llm_call` calls `LLMRanker.rerank` with `candidate_tasks=[]` and asserts both `result == []` and `llm.ainvoke.assert_not_called()`. The second assertion is the load-bearing one: it documents that the short-circuit is deliberate (avoid paying LLM token cost for a degenerate case) and that no network call is made in the empty-candidate path, which matters for tests and for production latency.

**OrcaNet benchmark test patterns:**

- *Controlled embedder for deterministic recall* — `test_retrieval_benchmark.py` bypasses CrossDomainEmbedder with a `_ControlledEmbedder` that maps each task's 25-dim feature vector to a pre-assigned orthonormal cluster embedding via a rounded-tuple lookup table. This ensures the FAISS similarity search is driven by known geometry (intra-group cosine ≈ 1, cross-group cosine ≈ 0) rather than learned representations, making the Recall@9 assertion deterministic and reproducible across seeds. `top_k_final=9` (one less than the group size of 10) prevents the self-hit from trivially filling the final slot and inflating recall to 1.0; ground truth excludes the query, so 8 of 9 positives retrieved yields a genuine Recall@9 ≈ 0.889 > 0.85.
- *Exact-cosine index without a FAISS binary* — `_SimpleFaissIndex` computes exact cosine similarity over all stored embeddings using numpy matrix operations, replacing the FAISS binary that is unavailable in CI. The index stores `(task_id, embedding)` pairs and returns top-k task IDs sorted by descending similarity. This lets the recall benchmark verify retrieval logic end-to-end without requiring a native FAISS install.
- *Orthonormal cluster centres for geometric separation* — tasks are partitioned into 10 groups using `np.eye(25)[g]` as the cluster centre. These are the standard basis vectors in ℝ²⁵, which are mutually orthogonal (cross-group cosine = 0) and unit-length (within-group cosine = 1 before noise). Adding `rng.standard_normal(25) * 0.02` per task perturbs each embedding slightly while preserving the inter-group separation that makes the recall assertion reliable.
- *Heterogeneous domain dataset construction* — the embedding benchmark builds five distinct data distributions (normal, correlated+skewed with exponential noise, sparse with 70% zeros, bimodal with ±4 means, heavy-tailed Student-t) to ensure the domain-invariance test is not trivially satisfied by random embeddings. Using structurally different feature spaces forces the DANN to learn genuinely domain-agnostic representations rather than collapsing on feature-space proximity.
- *DANN-vs-NeuralEmbedder gap comparison* — `test_dann_gap_smaller_than_contrastive_gap` is a relative assertion that does not hard-code the NeuralEmbedder gap. NeuralEmbedder uses NT-Xent contrastive loss, which clusters same-domain tasks together — deliberately increasing the domain gap. DANN uses gradient reversal, which pushes domain-invariance. The relative comparison enforces the architectural intent without requiring the NeuralEmbedder gap to be a specific value.
- *Binary label construction for Spearman* — `test_transfer_quality.py` uses `deepcopy(model)` for high-transfer pairs (CKA ≈ 1.0) and independently seeded random MLPs for low-transfer pairs (CKA < 0.5). The binary 1.0/0.0 label distribution is ideal for Spearman rank correlation: the two groups are well-separated in both CKA and label space, so the ρ > 0.60 assertion passes with margin even for shallow three-layer networks where the absolute CKA values shift with architecture.
- *Shared probe data across all pairs* — `_PROBE_DATA` is a fixed `(64, 8)` matrix seeded with `np.random.default_rng(0)`, shared across all 20 model pairs. Using the same probe data makes CKA scores directly comparable across pairs (CKA depends on both the model activations and the input distribution) and eliminates probe-data variance as a source of ranking noise.

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
uv run pytest packages/orcalab/tests/unit/test_dockerfile.py -v

# docker-compose.dev.yml service configuration
uv run pytest packages/orcalab/tests/unit/test_docker_compose.py -v

# Prefect work-pool initialisation script
uv run pytest packages/orcalab/tests/unit/test_init_prefect.py -v

# Module-level app export (uvicorn entrypoint)
uv run pytest packages/orcalab/tests/unit/test_app_module_export.py -v

# All deployment tests together
uv run pytest packages/orcalab/tests/unit/test_dockerfile.py \
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

# OrcaNet

> The cross-domain knowledge transfer agent of the [Orca](../../README.md) platform. Codename: The Connector.

---

OrcaNet answers a specific question: given a new ML task, which past tasks and models can accelerate it through transfer learning? It finds transferable source tasks using domain-adversarial embeddings and hybrid FAISS retrieval, scores transfer viability through feature-level and weight-level analysis, executes the transfer, and explains its reasoning through an LLM-powered agent. It coordinates with [OrcaMind](../orcamind/README.md) (source retrieval, model lookup) and [OrcaLab](../orcalab/README.md) (validation experiments) to run end-to-end transfer workflows.

## Layout

```text
orcanet/
├── embeddings/       Task representation learning (DANN, text, architecture GNN)
├── transfer/         Transfer strategies (feature CKA, weight, architecture, multi-task)
├── retrieval/        Hybrid 3-stage retrieval pipeline (FAISS → batch fetch → LLM re-ranking)
├── reasoning/        LangChain ReAct agent with 4 tools and structured output validation
│   ├── prompts/      Prompt templates (transfer explanation, similarity, architecture recommendation)
│   └── tools/        Agent tools (task retrieval, embedding similarity, transfer scoring, performance prediction)
├── integration/      End-to-end pipeline (OrcaNet → OrcaMind → OrcaLab)
├── api/              FastAPI REST service, 8 endpoints across 4 routers (port 8002)
├── cli.py            Typer CLI: serve, version
├── config/           Hydra YAML configs (LLM, retriever, embedder)
└── notebooks/        Interactive cross-domain transfer demo
```

### Embeddings

Three embedding strategies for representing ML tasks in vector space:

| Embedder | Approach | Output Dim | Reference |
|----------|----------|-----------|-----------|
| CrossDomainEmbedder | Domain-Adversarial Neural Network (DANN) with gradient reversal | 64 | Ganin et al. 2016 |
| TextTaskEmbedder | sentence-transformers text encoding fused with statistical features | configurable | — |
| ArchitectureEmbedder | GNN-based encoding of model architecture graphs | 32 | — |

`CrossDomainEmbedder` maps 25-dimensional statistical meta-features to a domain-invariant embedding space through a shared trunk, task-type classifier, and adversarial domain classifier. The gradient reversal layer forces the embedding to be informative for task classification while remaining invariant to domain, which is what makes cross-domain transfer scoring work.

`TextTaskEmbedder` encodes natural language task descriptions via `all-MiniLM-L6-v2` and fuses them with statistical features using concat, add, or attention fusion. `ArchitectureEmbedder` converts model config dicts into graph-structured tensors and embeds them via residual adjacency-matrix message passing.

All embedders produce L2-normalised output for cosine similarity retrieval. The `orcanet.embeddings` module uses a lazy `__getattr__` shim to defer `import torch` until first access.

### Transfer Strategies

Four strategies, all implementing the `TransferStrategy` ABC (`score_transfer` / `execute_transfer` / `get_transfer_metadata`):

| Strategy | Scoring Method | Transfer Method |
|----------|---------------|-----------------|
| FeatureTransfer | Linear CKA (Kornblith et al. 2019) with depth weighting | Clone target, patch matched-layer weights |
| WeightTransfer | Parameter matching by name, shape, or both | Deep-copy target, transfer matched weights, reinitialise unmatched |
| ArchitectureTransfer | Graph-embedding cosine similarity via OrcaMind | Adapt architecture dims, copy middle-layer weights |
| MultiTaskTransfer | CrossDomainEmbedder cosine similarity | Joint training with equal, uncertainty (Kendall et al. 2018), or GradNorm weighting |

Each strategy returns a `TransferScore` with an overall score, per-layer scores, recommended layers, and human-readable reasoning.

### Retrieval

A three-stage async pipeline that narrows from broad vector similarity to a small set of semantically ranked candidates:

1. FAISS vector search: embed the query task, retrieve `top_k_initial` (default 50) nearest neighbours from the pre-built FAISS index.
2. Batch fetch and filter: resolve task IDs from the registry, apply similarity threshold and optional field filters, skip failed fetches without aborting.
3. LLM re-ranking: when candidates exceed `top_k_final` (default 10), delegate to `LLMRanker` for semantic re-ranking against the query task.

`QueryExpander` generates multiple reformulations of the query via an LLM, driving distinct FAISS embeddings per expansion. Results are deduplicated by task ID (keeping highest score) and returned sorted.

### Reasoning

`OrcaNetAgent` wraps a LangChain `CompiledStateGraph` (ReAct pattern) with four async tools:

| Tool | Function |
|------|----------|
| `task_retrieval_tool` | Find similar tasks via `HybridRetriever` |
| `embedding_similarity_tool` | Compute pairwise cosine similarity |
| `transfer_scoring_tool` | Score a named transfer strategy |
| `performance_prediction_tool` | Predict target performance via OrcaMind |

The agent produces a `TransferRecommendationResponse` (Pydantic v2 validated) with top source tasks, recommended strategy, expected improvement, confidence, and natural-language explanation. Retry logic re-invokes the agent up to 3 times on validation failures. LLM support covers OpenAI, Anthropic, and local OpenAI-compatible endpoints.

## API

FastAPI service on port 8002 with 8 endpoints:

| Router | Key Endpoints |
|--------|--------------|
| Transfer | Score transfer viability, execute transfer |
| Retrieve | Find similar tasks with optional LLM re-ranking |
| Embed | Generate task embeddings |
| Explain | LLM-powered transfer explanations |

Interactive docs at `http://localhost:8002/docs`. Full spec in [API Reference](../../docs/API-REFERENCE.md).

## CLI

```bash
orcanet serve            # Start the FastAPI server (--reload for dev)
orcanet version          # Print package version
```

## Configuration

Hydra YAML configs under `config/`:

```text
config/
├── config.yaml                Root config (LLM, retrieval thresholds, OrcaMind/OrcaLab URLs)
├── embedder/cross_domain.yaml DANN dimensions (input=25, embedding=64, n_domains=10)
├── retriever/hybrid.yaml      FAISS index path, top-k thresholds, similarity threshold
└── llm/openai.yaml            LLM provider, model, temperature
```

OrcaNet supports three LLM backends via the `ORCANET_LLM_PROVIDER` environment variable: `openai`, `anthropic`, and `local`. Variable reference in [Deployment](../../docs/DEPLOYMENT.md).

## Integration Points

OrcaNet coordinates with the other two services for end-to-end knowledge transfer:

| Direction | Mechanism |
|-----------|-----------|
| OrcaNet → OrcaMind | Retrieve best models for source tasks; recommend architectures for target tasks |
| OrcaNet → OrcaLab | Dispatch validation experiments when `transfer_score > 0.4` |
| OrcaLab → OrcaNet | Validated metrics written back to `transfer_mappings` for future queries |

OrcaLab calls are guarded by timeouts and degrade gracefully: if validation times out the transfer mapping is stored with `experiment_result=None`. If OrcaMind is unreachable (`httpx.ConnectError` or `httpx.TimeoutException`) the transfer pipeline raises `ServiceUnavailableError` and the API returns a 503.

Integration diagram in [Architecture](../../docs/ARCHITECTURE.md).

## Testing

```bash
pytest packages/orcanet/tests/unit        # 200+ unit tests
pytest packages/orcanet/tests/benchmarks  # Recall, embedding quality, transfer quality
```

The test suite covers embedding strategies, transfer strategies, retrieval stages, reasoning agent tools and validation, and API integration. Offline stubs (deterministic sentence-transformer, mocked LLMs) let the full suite run without network access. More detail in [Development](../../docs/DEVELOPMENT.md).

## Tech Stack

| Category | Libraries |
|----------|-----------|
| Embeddings | PyTorch, sentence-transformers, FAISS |
| Transfer | PyTorch (CKA, weight matching, multi-task) |
| GNN | torch-geometric (optional: `pip install orcanet[gnn]`) |
| Reasoning | LangChain, langchain-openai, langchain-anthropic |
| API | FastAPI, Uvicorn, Pydantic v2 |
| Config | Hydra, OmegaConf |
| CLI | Typer |
| Data | NumPy, SciPy, scikit-learn, pandas |
| Shared | [orca-shared](../orca-shared/README.md), [orcamind](../orcamind/README.md), [orcalab](../orcalab/README.md) |

---

[Back to packages](../README.md)

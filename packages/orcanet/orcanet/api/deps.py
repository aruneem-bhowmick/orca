"""FastAPI dependency providers backed by app.state singletons."""

from __future__ import annotations

import logging
import os
from typing import AsyncGenerator

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from orca_shared.clients.orcalab_client import OrcaLabClient
from orca_shared.clients.orcamind_client import OrcaMindClient
from orca_shared.registry.repository import TaskRepository
from orcanet.embeddings.cross_domain import CrossDomainEmbedder
from orcanet.integration.pipeline import TransferPipeline
from orcanet.reasoning.agent import OrcaNetAgent
from orcanet.retrieval.retriever import HybridRetriever
from orcanet.transfer.base import TransferStrategy

logger = logging.getLogger("orcanet.api")


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session scoped to the current request transaction."""
    async with request.app.state.db_sessionmaker() as session:
        async with session.begin():
            yield session


async def get_task_repo(
    session: AsyncSession = Depends(get_db),
) -> TaskRepository:
    """Return a :class:`~orca_shared.registry.repository.TaskRepository` for the current session."""
    return TaskRepository(session)


def get_hybrid_retriever(request: Request) -> HybridRetriever:
    """Return the singleton :class:`~orcanet.retrieval.retriever.HybridRetriever`.

    Raises 503 if the retriever was not initialised during application startup.
    """
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")
    return retriever


def get_cross_domain_embedder(request: Request) -> CrossDomainEmbedder:
    """Return the singleton :class:`~orcanet.embeddings.cross_domain.CrossDomainEmbedder`."""
    return request.app.state.embedder


def get_orcamind_client(request: Request) -> OrcaMindClient:
    """Return the singleton :class:`~orca_shared.clients.orcamind_client.OrcaMindClient`."""
    return request.app.state.orcamind_client


def get_orcalab_client(request: Request) -> OrcaLabClient:
    """Return the singleton :class:`~orca_shared.clients.orcalab_client.OrcaLabClient`."""
    return request.app.state.orcalab_client


def get_transfer_strategies(request: Request) -> dict[str, TransferStrategy]:
    """Return the transfer-strategy registry mapping strategy name to implementation."""
    return request.app.state.transfer_strategies


def get_transfer_pipeline(
    request: Request,
    task_repo: TaskRepository = Depends(get_task_repo),
) -> TransferPipeline:
    """Build a :class:`~orcanet.integration.pipeline.TransferPipeline` for the current request."""
    return TransferPipeline(
        orcamind_client=request.app.state.orcamind_client,
        orcalab_client=request.app.state.orcalab_client,
        transfer_strategies=request.app.state.transfer_strategies,
        task_repository=task_repo,
    )


_ALLOWED_LLM_PROVIDERS = frozenset({"openai", "anthropic", "local"})


async def get_orcanet_agent(
    request: Request,
    task_repo: TaskRepository = Depends(get_task_repo),
) -> OrcaNetAgent:
    """Return the shared agent, or a fresh one for per-request provider overrides.

    Reads ``X-LLM-Provider`` header. If set, the value must be one of
    ``openai``, ``anthropic``, or ``local``; any other value raises 400.
    The override agent receives the same request-scoped ``TaskRepository``
    as the shared agent so all tool capabilities remain intact.
    """
    header_provider = request.headers.get("X-LLM-Provider", "").strip()
    if header_provider:
        if header_provider not in _ALLOWED_LLM_PROVIDERS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"X-LLM-Provider {header_provider!r} is not supported. "
                    f"Allowed values: {sorted(_ALLOWED_LLM_PROVIDERS)}"
                ),
            )
        logger.debug("X-LLM-Provider override: %s", header_provider)
        return OrcaNetAgent(
            llm_provider=header_provider,
            api_key=os.environ.get("ORCANET_LLM_API_KEY"),
            retriever=getattr(request.app.state, "retriever", None),
            embedder=getattr(request.app.state, "embedder", None),
            task_repository=task_repo,
            transfer_strategies=request.app.state.transfer_strategies,
            orcamind_client=request.app.state.orcamind_client,
        )
    return request.app.state.agent

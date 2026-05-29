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
from orcanet.reasoning.agent import OrcaNetAgent
from orcanet.retrieval.retriever import HybridRetriever
from orcanet.transfer.base import TransferStrategy

logger = logging.getLogger("orcanet.api")


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    async with request.app.state.db_sessionmaker() as session:
        async with session.begin():
            yield session


async def get_task_repo(
    session: AsyncSession = Depends(get_db),
) -> TaskRepository:
    return TaskRepository(session)


def get_hybrid_retriever(request: Request) -> HybridRetriever:
    retriever = request.app.state.retriever
    if retriever is None:
        raise HTTPException(status_code=503, detail="Retriever not initialised")
    return retriever


def get_cross_domain_embedder(request: Request) -> CrossDomainEmbedder:
    return request.app.state.embedder


def get_orcamind_client(request: Request) -> OrcaMindClient:
    return request.app.state.orcamind_client


def get_orcalab_client(request: Request) -> OrcaLabClient:
    return request.app.state.orcalab_client


def get_transfer_strategies(request: Request) -> dict[str, TransferStrategy]:
    return request.app.state.transfer_strategies


def get_orcanet_agent(request: Request) -> OrcaNetAgent:
    """Return the shared agent, or a fresh one for per-request provider overrides.

    Reads ``X-LLM-Provider`` header. If set, constructs a new ``OrcaNetAgent``
    using that provider and the API key from the environment so that individual
    requests can target different LLM backends without restarting the service.
    """
    header_provider = request.headers.get("X-LLM-Provider", "").strip()
    if header_provider:
        logger.debug("X-LLM-Provider override: %s", header_provider)
        return OrcaNetAgent(
            llm_provider=header_provider,
            api_key=os.environ.get("ORCANET_LLM_API_KEY"),
            retriever=getattr(request.app.state, "retriever", None),
            embedder=getattr(request.app.state, "embedder", None),
            task_repository=None,
            transfer_strategies=request.app.state.transfer_strategies,
            orcamind_client=request.app.state.orcamind_client,
        )
    return request.app.state.agent

"""FastAPI dependency providers backed by app.state singletons."""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends, HTTPException, Request
from sqlalchemy.ext.asyncio import AsyncSession

from orca_shared.registry.repository import (
    EmbeddingRepository,
    ExperimentRepository,
    PerformanceRepository,
    TaskRepository,
)
from orcamind.embedders.similarity import FaissIndex
from orcamind.embedders.statistical import StatisticalEmbedder
from orcamind.selectors.nearest_neighbor import NearestNeighborSelector
from orcamind.selectors.predictor import PerformancePredictor


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    async with request.app.state.db_sessionmaker() as session:
        async with session.begin():
            yield session


async def get_task_repo(
    session: AsyncSession = Depends(get_db),
) -> TaskRepository:
    return TaskRepository(session)


async def get_experiment_repo(
    session: AsyncSession = Depends(get_db),
) -> ExperimentRepository:
    return ExperimentRepository(session)


async def get_embedding_repo(
    session: AsyncSession = Depends(get_db),
) -> EmbeddingRepository:
    return EmbeddingRepository(session)


async def get_perf_repo(
    session: AsyncSession = Depends(get_db),
) -> PerformanceRepository:
    return PerformanceRepository(session)


def get_faiss_index(request: Request) -> FaissIndex:
    idx = request.app.state.faiss_index
    if idx is None:
        raise HTTPException(status_code=503, detail="FAISS index not loaded")
    return idx


def get_stat_embedder(request: Request) -> StatisticalEmbedder:
    return request.app.state.stat_embedder


def get_nn_selector(request: Request) -> NearestNeighborSelector:
    return request.app.state.nn_selector


def get_predictor(request: Request) -> PerformancePredictor:
    return request.app.state.predictor

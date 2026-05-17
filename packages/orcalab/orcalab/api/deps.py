"""FastAPI dependency providers backed by app.state singletons."""

from __future__ import annotations

from typing import AsyncGenerator

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from orca_shared.registry.repository import ExperimentRepository, SearchSpaceRepository


async def get_db(request: Request) -> AsyncGenerator[AsyncSession, None]:
    async with request.app.state.db_sessionmaker() as session:
        async with session.begin():
            yield session


async def get_experiment_repo(
    session: AsyncSession = Depends(get_db),
) -> ExperimentRepository:
    return ExperimentRepository(session)


async def get_search_space_repo(
    session: AsyncSession = Depends(get_db),
) -> SearchSpaceRepository:
    return SearchSpaceRepository(session)


def get_sweeps_store(request: Request) -> dict:
    return request.app.state.sweeps

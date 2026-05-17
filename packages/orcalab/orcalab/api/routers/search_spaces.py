"""Search space persistence endpoints."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, status
from pydantic import BaseModel

from orca_shared.registry.repository import SearchSpaceRepository
from orca_shared.schemas.search_space import SearchSpaceRecord

from ..deps import get_search_space_repo

router = APIRouter(prefix="/search-spaces", tags=["search-spaces"])


class CreateSearchSpaceRequest(BaseModel):
    name: str | None = None
    parameters: list[dict[str, Any]] = []
    description: str = ""


@router.post("", response_model=SearchSpaceRecord, status_code=status.HTTP_201_CREATED)
async def create_search_space(
    body: CreateSearchSpaceRequest,
    repo: SearchSpaceRepository = Depends(get_search_space_repo),
) -> SearchSpaceRecord:
    definition = {
        "name": body.name,
        "description": body.description,
        "parameters": body.parameters,
    }
    return await repo.create(name=body.name, definition=definition)


@router.get("", response_model=list[SearchSpaceRecord])
async def list_search_spaces(
    limit: int = Query(default=50, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    repo: SearchSpaceRepository = Depends(get_search_space_repo),
) -> list[SearchSpaceRecord]:
    return await repo.list_all(limit=limit, offset=offset)

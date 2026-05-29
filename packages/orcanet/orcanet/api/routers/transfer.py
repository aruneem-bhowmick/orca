"""Transfer scoring, recommendation, validation, and mapping lookup endpoints."""

from __future__ import annotations

import logging
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from orca_shared.registry.models import TransferMapping as TransferMappingORM
from orca_shared.registry.repository import TaskRepository
from orca_shared.schemas.transfer import TransferMapping
from orcanet.api.deps import (
    get_db,
    get_orcanet_agent,
    get_task_repo,
    get_transfer_pipeline,
    get_transfer_strategies,
)
from orcanet.api.schemas import (
    TransferRecommendRequest,
    TransferScoreRequest,
    TransferScoreResponse,
    TransferValidateRequest,
    TransferValidateResponse,
)
from orcanet.integration.pipeline import ServiceUnavailableError, TransferPipeline
from orcanet.reasoning.agent import OrcaNetAgent
from orcanet.reasoning.validators import TransferRecommendationResponse
from orcanet.transfer.base import TransferStrategy

logger = logging.getLogger("orcanet.api")

router = APIRouter(prefix="/transfer", tags=["transfer"])


@router.post("/score")
async def score_transfer(
    body: TransferScoreRequest,
    task_repo: TaskRepository = Depends(get_task_repo),
    strategies: dict[str, TransferStrategy] = Depends(get_transfer_strategies),
) -> dict:
    """Score the transferability between two tasks."""
    strategy_name = body.strategy
    if strategy_name not in strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown strategy {strategy_name!r}. Valid: {sorted(strategies)}",
        )

    try:
        source_uuid = UUID(body.source_task_id)
        target_uuid = UUID(body.target_task_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    source_task = await task_repo.get_by_id(source_uuid)
    if source_task is None:
        raise HTTPException(status_code=404, detail=f"Source task {body.source_task_id} not found")

    target_task = await task_repo.get_by_id(target_uuid)
    if target_task is None:
        raise HTTPException(status_code=404, detail=f"Target task {body.target_task_id} not found")

    score = strategies[strategy_name].score_transfer(source_task, target_task)
    return {
        "overall": score.overall,
        "layer_scores": score.layer_scores,
        "recommended_layers": score.recommended_layers,
        "reasoning": score.reasoning,
        "strategy": strategy_name,
    }


@router.post("/recommend", response_model=TransferRecommendationResponse)
async def recommend_transfer(
    body: TransferRecommendRequest,
    agent: OrcaNetAgent = Depends(get_orcanet_agent),
) -> TransferRecommendationResponse:
    """Run the OrcaNet reasoning agent to recommend transfer sources."""
    query = (
        f"Recommend transfer sources for target task {body.target_task_id}. "
        f"Description: {body.query_description}. Return top {body.top_k} sources."
    )
    try:
        return await agent.recommend_transfer(query)
    except Exception as exc:
        logger.error("Agent failed during recommend_transfer: %s", exc)
        raise HTTPException(status_code=502, detail="LLM agent error") from exc


@router.post("/validate", response_model=TransferValidateResponse)
async def validate_transfer(
    body: TransferValidateRequest,
    pipeline: TransferPipeline = Depends(get_transfer_pipeline),
) -> TransferValidateResponse:
    """Run the full three-way transfer pipeline and persist the result.

    Scores the transfer, optionally triggers an OrcaLab validation experiment,
    and saves a transfer mapping to the database regardless of the outcome.
    """
    try:
        result = await pipeline.recommend_and_validate(
            source_task_id=body.source_task_id,
            target_task_id=body.target_task_id,
            strategy_name=body.strategy,
            validate=body.run_validation,
        )
    except ServiceUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {exc}") from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return TransferValidateResponse(
        score=TransferScoreResponse(
            overall=result.score.overall,
            layer_scores=result.score.layer_scores,
            recommended_layers=result.score.recommended_layers,
            reasoning=result.score.reasoning,
        ),
        experiment_result=result.experiment_result,
        mapping=result.mapping,
        improvement_over_baseline=result.improvement_over_baseline,
    )


@router.get("/{mapping_id}", response_model=TransferMapping)
async def get_transfer_mapping(
    mapping_id: str,
    session: AsyncSession = Depends(get_db),
) -> TransferMapping:
    """Retrieve a stored transfer mapping by its ID."""
    try:
        mapping_uuid = UUID(mapping_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = await session.execute(
        select(TransferMappingORM).where(TransferMappingORM.mapping_id == mapping_uuid)
    )
    row = result.scalar_one_or_none()
    if row is None:
        raise HTTPException(status_code=404, detail=f"TransferMapping {mapping_id} not found")

    return TransferMapping.model_validate(row)

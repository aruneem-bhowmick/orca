"""LLM-powered transfer explanation endpoint."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException

from orcanet.api.deps import get_orcanet_agent
from orcanet.api.schemas import ExplainRequest, ExplainResponse
from orcanet.reasoning.agent import OrcaNetAgent
from orcanet.reasoning.validators import LLMParsingError

logger = logging.getLogger("orcanet.api")

router = APIRouter(tags=["explain"])


@router.post("/explain", response_model=ExplainResponse)
async def explain_transfer(
    body: ExplainRequest,
    agent: OrcaNetAgent = Depends(get_orcanet_agent),
) -> ExplainResponse:
    """Generate a human-readable explanation for a specific transfer."""
    query = (
        f"Explain the transfer from source task {body.source_task_id} "
        f"to target task {body.target_task_id} using the {body.strategy} strategy. "
        "Describe why this transfer is or is not likely to be beneficial."
    )
    try:
        response = await agent.recommend_transfer(query)
        return ExplainResponse(explanation=response.explanation)
    except LLMParsingError as exc:
        logger.error("LLM parsing failed during explain: %s", exc)
        raise HTTPException(
            status_code=502, detail="LLM agent returned an unparseable response"
        ) from exc
    except Exception as exc:
        logger.error("Agent error during explain: %s", exc)
        raise HTTPException(status_code=502, detail="LLM agent error") from exc

"""Pydantic response schemas and error types for the OrcaNet reasoning agent."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

TransferStrategy = Literal["feature", "weight", "architecture", "multi_task"]


class LLMParsingError(Exception):
    """Raised when the LLM output cannot be parsed into a valid response after all retries."""


class SourceTaskRecommendation(BaseModel):
    """A single source-task recommendation with transfer scoring detail."""

    task_id: str
    task_name: str
    similarity_score: float = Field(ge=0.0, le=1.0)
    transfer_score: float = Field(ge=0.0, le=1.0)
    reasoning: str


class TransferRecommendationResponse(BaseModel):
    """Structured output returned by OrcaNetAgent.recommend_transfer()."""

    top_sources: list[SourceTaskRecommendation]
    recommended_strategy: TransferStrategy
    expected_improvement: float = Field(ge=0.0, le=1.0)
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)

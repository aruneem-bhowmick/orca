"""Prompt templates for the OrcaNet reasoning agent."""

from orcanet.reasoning.prompts.architecture_recommendation import (
    ARCHITECTURE_RECOMMENDATION_TEMPLATE,
)
from orcanet.reasoning.prompts.task_similarity import TASK_SIMILARITY_TEMPLATE
from orcanet.reasoning.prompts.transfer_explanation import TRANSFER_EXPLANATION_TEMPLATE

__all__ = [
    "TRANSFER_EXPLANATION_TEMPLATE",
    "TASK_SIMILARITY_TEMPLATE",
    "ARCHITECTURE_RECOMMENDATION_TEMPLATE",
]

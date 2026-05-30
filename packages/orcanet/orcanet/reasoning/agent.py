"""OrcaNet reasoning agent for transfer learning recommendations."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage
from pydantic import ValidationError

from orcanet.reasoning.tools.embedding_similarity_tool import make_embedding_similarity_tool
from orcanet.reasoning.tools.performance_prediction_tool import make_performance_prediction_tool
from orcanet.reasoning.tools.task_retrieval_tool import make_task_retrieval_tool
from orcanet.reasoning.tools.transfer_scoring_tool import make_transfer_scoring_tool
from orcanet.reasoning.validators import LLMParsingError, TransferRecommendationResponse

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2

_SYSTEM_PROMPT = """\
You are OrcaNet, an expert in cross-domain transfer learning. You help users identify the
best source tasks and strategies for knowledge transfer to a target task.

When you have gathered enough information using the available tools, respond with a JSON
object that exactly matches this schema (no markdown fences, no extra text):
{
  "top_sources": [
    {
      "task_id": "<uuid>",
      "task_name": "<string>",
      "similarity_score": <float 0-1>,
      "transfer_score": <float 0-1>,
      "reasoning": "<string>"
    }
  ],
  "recommended_strategy": "<feature|weight|architecture|multi_task>",
  "expected_improvement": <float 0-1>,
  "explanation": "<string>",
  "confidence": <float 0-1>
}
"""

_CORRECTIVE_SUFFIX = (
    "\n\nYour previous response could not be parsed as valid JSON. "
    "Please respond ONLY with a valid JSON object matching the required schema. "
    "No markdown fences, no extra text."
)


class OrcaNetAgent:
    """LangChain agent for transfer learning recommendations.

    Uses the LangChain 1.x ``create_agent`` API backed by langgraph.
    The ``recommend_transfer`` method retries parsing up to ``_MAX_RETRIES``
    times with a corrective prompt before raising ``LLMParsingError``.
    """

    def __init__(
        self,
        llm_provider: str = "openai",
        model: str = "gpt-4-turbo",
        temperature: float = 0.7,
        api_key: str | None = None,
        *,
        retriever: Any = None,
        embedder: Any = None,
        task_repository: Any = None,
        transfer_strategies: dict | None = None,
        orcamind_client: Any = None,
    ) -> None:
        """Build the agent: wire tools, construct the LLM, and create the langgraph agent.

        Args:
            llm_provider: Backend provider — ``"openai"``, ``"anthropic"``, or ``"local"``.
            model: Model identifier forwarded to the provider (e.g. ``"gpt-4-turbo"``).
            temperature: Sampling temperature for LLM output.
            api_key: Provider API key; falls back to provider-specific env var when ``None``.
            retriever: :class:`~orcanet.retrieval.retriever.HybridRetriever` instance for
                the task-retrieval tool.
            embedder: :class:`~orcanet.embeddings.cross_domain.CrossDomainEmbedder` for
                the embedding-similarity tool.
            task_repository: :class:`~orca_shared.registry.repository.TaskRepository` used
                by retrieval, similarity, and scoring tools.
            transfer_strategies: Mapping of strategy name → :class:`TransferStrategy` for
                the transfer-scoring tool.
            orcamind_client: :class:`~orca_shared.clients.orcamind_client.OrcaMindClient`
                for the performance-prediction tool.
        """
        self.tools = [
            make_task_retrieval_tool(retriever),
            make_embedding_similarity_tool(embedder, task_repository),
            make_transfer_scoring_tool(transfer_strategies or {}, task_repository),
            make_performance_prediction_tool(orcamind_client, task_repository),
        ]
        self.llm = self._build_llm(llm_provider, model, temperature, api_key)
        self._agent = create_agent(
            self.llm,
            self.tools,
            system_prompt=_SYSTEM_PROMPT,
        )

    async def recommend_transfer(self, query: str) -> TransferRecommendationResponse:
        """Run the agent loop and return a validated transfer recommendation.

        Retries up to ``_MAX_RETRIES`` times with a corrective prompt when the
        agent's final message cannot be parsed into a ``TransferRecommendationResponse``.
        Raises ``LLMParsingError`` after all attempts are exhausted.
        """
        messages: list = [HumanMessage(content=query)]
        last_output: str = ""

        for attempt in range(_MAX_RETRIES + 1):
            result = await self._agent.ainvoke({"messages": messages})
            last_message = result["messages"][-1]
            last_output = (
                last_message.content
                if isinstance(last_message.content, str)
                else str(last_message.content)
            )
            try:
                return self._parse_and_validate(last_output)
            except (ValidationError, json.JSONDecodeError, ValueError) as exc:
                if attempt == _MAX_RETRIES:
                    raise LLMParsingError(
                        f"Failed to parse LLM output after {_MAX_RETRIES + 1} attempts. "
                        f"Last output: {last_output!r}"
                    ) from exc
                logger.warning(
                    "OrcaNetAgent: parse attempt %d/%d failed: %s",
                    attempt + 1,
                    _MAX_RETRIES + 1,
                    exc,
                )
                messages = [HumanMessage(content=query + _CORRECTIVE_SUFFIX)]

        raise LLMParsingError("Unreachable")  # pragma: no cover

    def _parse_and_validate(self, output: str) -> TransferRecommendationResponse:
        """Strip markdown fences and validate as TransferRecommendationResponse."""
        text = output.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3].strip()
        return TransferRecommendationResponse.model_validate_json(text)

    def _build_llm(self, provider: str, model: str, temperature: float, api_key: str | None):
        """Instantiate and return the LangChain chat model for *provider*.

        Raises ``ValueError`` for unknown provider strings.
        """
        if provider == "openai":
            from langchain_openai import ChatOpenAI

            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
        if provider == "anthropic":
            from langchain_anthropic import ChatAnthropic

            return ChatAnthropic(
                model=model,
                temperature=temperature,
                api_key=api_key,
            )
        if provider == "local":
            from langchain_openai import ChatOpenAI

            base_url = os.environ.get("ORCANET_LOCAL_LLM_URL", "http://localhost:11434/v1")
            return ChatOpenAI(
                model=model,
                temperature=temperature,
                api_key=api_key or "local",
                base_url=base_url,
            )
        raise ValueError(
            f"Unknown LLM provider {provider!r}. "
            "Supported providers: 'openai', 'anthropic', 'local'."
        )

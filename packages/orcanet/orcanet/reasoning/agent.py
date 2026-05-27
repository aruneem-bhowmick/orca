"""OrcaNet ReAct reasoning agent for transfer learning recommendations."""

from __future__ import annotations

import json
import logging
import os
from typing import Any

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from orcanet.reasoning.tools import (
    embedding_similarity_tool,
    performance_prediction_tool,
    task_retrieval_tool,
    transfer_scoring_tool,
)
from orcanet.reasoning.tools import embedding_similarity_tool as _est_mod
from orcanet.reasoning.tools import performance_prediction_tool as _ppt_mod
from orcanet.reasoning.tools import task_retrieval_tool as _trt_mod
from orcanet.reasoning.tools import transfer_scoring_tool as _tst_mod
from orcanet.reasoning.validators import LLMParsingError, TransferRecommendationResponse

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2

_REACT_SYSTEM = """\
You are OrcaNet, an expert in cross-domain transfer learning. You help users find the best \
source tasks and strategies for knowledge transfer.

You have access to the following tools:

{tools}

Use this format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: a JSON object matching this schema:
{{
  "top_sources": [
    {{
      "task_id": "<uuid>",
      "task_name": "<string>",
      "similarity_score": <float 0-1>,
      "transfer_score": <float 0-1>,
      "reasoning": "<string>"
    }}
  ],
  "recommended_strategy": "<feature|weight|architecture|multi_task>",
  "expected_improvement": <float 0-1>,
  "explanation": "<string>",
  "confidence": <float 0-1>
}}

Begin!
"""

_REACT_HUMAN = """\
Question: {input}
Thought:{agent_scratchpad}"""

_CORRECTIVE_SUFFIX = (
    "\n\nYour previous response could not be parsed as valid JSON. "
    "Please respond ONLY with a valid JSON object matching the required schema. "
    "No markdown fences, no extra text."
)


class OrcaNetAgent:
    """LangChain ReAct agent for transfer learning recommendations.

    Wraps four LangChain tools and an LLM into a ReAct loop. The
    ``recommend_transfer`` method retries parsing up to two times
    with a corrective prompt before raising ``LLMParsingError``.
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
        import orcanet.reasoning.tools.embedding_similarity_tool as _em
        import orcanet.reasoning.tools.performance_prediction_tool as _pp
        import orcanet.reasoning.tools.task_retrieval_tool as _tr
        import orcanet.reasoning.tools.transfer_scoring_tool as _ts

        if retriever is not None:
            _tr.set_retriever(retriever)
        if embedder is not None:
            _em.set_embedder(embedder)
        if task_repository is not None:
            _em.set_task_repository(task_repository)
            _ts.set_task_repository(task_repository)
            _pp.set_task_repository(task_repository)
        if transfer_strategies is not None:
            _ts.set_transfer_strategies(transfer_strategies)
        if orcamind_client is not None:
            _pp.set_orcamind_client(orcamind_client)

        self.tools = [
            task_retrieval_tool,
            embedding_similarity_tool,
            transfer_scoring_tool,
            performance_prediction_tool,
        ]
        self.llm = self._build_llm(llm_provider, model, temperature, api_key)
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", _REACT_SYSTEM),
                ("human", _REACT_HUMAN),
            ]
        )
        agent = create_react_agent(self.llm, self.tools, prompt)
        self.executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
        )

    async def recommend_transfer(self, query: str) -> TransferRecommendationResponse:
        """Run the ReAct loop and return a validated transfer recommendation.

        Retries up to ``_MAX_RETRIES`` times with a corrective prompt if the
        LLM output cannot be parsed.  Raises ``LLMParsingError`` after all
        attempts are exhausted.
        """
        current_query = query
        last_output: str = ""
        for attempt in range(_MAX_RETRIES + 1):
            raw = await self.executor.ainvoke({"input": current_query})
            last_output = raw.get("output", "")
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
                current_query = query + _CORRECTIVE_SUFFIX

        raise LLMParsingError("Unreachable")  # pragma: no cover

    def _parse_and_validate(self, output: str) -> TransferRecommendationResponse:
        """Extract and validate a TransferRecommendationResponse from raw LLM output."""
        text = output.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            text = "\n".join(lines[1:])
            if text.endswith("```"):
                text = text[:-3].strip()
        return TransferRecommendationResponse.model_validate_json(text)

    def _build_llm(self, provider: str, model: str, temperature: float, api_key: str | None):
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
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            model=model,
            temperature=temperature,
            api_key=api_key,
        )

"""Unit tests for OrcaNetAgent."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from orcanet.reasoning.validators import (
    LLMParsingError,
    SourceTaskRecommendation,
    TransferRecommendationResponse,
)

_VALID_RESPONSE_JSON = json.dumps(
    {
        "top_sources": [
            {
                "task_id": "11111111-1111-1111-1111-111111111111",
                "task_name": "brain MRI classification",
                "similarity_score": 0.88,
                "transfer_score": 0.75,
                "reasoning": "Shared convolutional feature hierarchy across medical imaging tasks.",
            }
        ],
        "recommended_strategy": "feature",
        "expected_improvement": 0.18,
        "explanation": "Feature-level transfer is beneficial due to high domain similarity.",
        "confidence": 0.82,
    }
)

_INVALID_JSON = "this is not valid json at all"

_FENCED_RESPONSE_JSON = f"```json\n{_VALID_RESPONSE_JSON}\n```"


def _build_agent(mock_ainvoke_return: list | None = None):
    """Build an OrcaNetAgent with all external dependencies mocked."""
    from langchain_core.messages import AIMessage

    if mock_ainvoke_return is None:
        mock_ainvoke_return = [{"messages": [AIMessage(content=_VALID_RESPONSE_JSON)]}]

    mock_agent_graph = MagicMock()
    mock_agent_graph.ainvoke = AsyncMock(side_effect=mock_ainvoke_return)

    mock_llm = MagicMock()
    mock_llm.bind_tools = MagicMock(return_value=mock_llm)

    with (
        patch("orcanet.reasoning.agent.create_agent", return_value=mock_agent_graph),
        patch("orcanet.reasoning.agent.OrcaNetAgent._build_llm", return_value=mock_llm),
    ):
        from orcanet.reasoning.agent import OrcaNetAgent

        agent = OrcaNetAgent.__new__(OrcaNetAgent)
        agent.tools = []
        agent.llm = mock_llm
        agent._agent = mock_agent_graph

    return agent, mock_agent_graph


class TestOrcaNetAgentTools:
    def test_agent_has_four_tools(self) -> None:
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        with (
            patch("orcanet.reasoning.agent.create_agent", return_value=MagicMock()),
            patch("langchain_openai.ChatOpenAI", return_value=mock_llm),
        ):
            from orcanet.reasoning.agent import OrcaNetAgent

            agent = OrcaNetAgent(api_key="fake")
            assert len(agent.tools) == 4

    def test_all_four_tool_names_present(self) -> None:
        mock_llm = MagicMock()
        mock_llm.bind_tools = MagicMock(return_value=mock_llm)

        with (
            patch("orcanet.reasoning.agent.create_agent", return_value=MagicMock()),
            patch("langchain_openai.ChatOpenAI", return_value=mock_llm),
        ):
            from orcanet.reasoning.agent import OrcaNetAgent

            agent = OrcaNetAgent(api_key="fake")
            tool_names = {t.name for t in agent.tools}
            assert "task_retrieval_tool" in tool_names
            assert "embedding_similarity_tool" in tool_names
            assert "transfer_scoring_tool" in tool_names
            assert "performance_prediction_tool" in tool_names


class TestRecommendTransfer:
    @pytest.mark.asyncio
    async def test_returns_transfer_recommendation_response(self) -> None:
        from langchain_core.messages import AIMessage

        agent, _ = _build_agent(
            [{"messages": [AIMessage(content=_VALID_RESPONSE_JSON)]}]
        )
        result = await agent.recommend_transfer("find similar tasks to brain MRI classification")
        assert isinstance(result, TransferRecommendationResponse)

    @pytest.mark.asyncio
    async def test_parses_top_sources_correctly(self) -> None:
        from langchain_core.messages import AIMessage

        agent, _ = _build_agent(
            [{"messages": [AIMessage(content=_VALID_RESPONSE_JSON)]}]
        )
        result = await agent.recommend_transfer("brain MRI classification")
        assert len(result.top_sources) == 1
        assert result.top_sources[0].task_name == "brain MRI classification"
        assert result.top_sources[0].similarity_score == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_strips_markdown_fences_before_parsing(self) -> None:
        from langchain_core.messages import AIMessage

        agent, _ = _build_agent(
            [{"messages": [AIMessage(content=_FENCED_RESPONSE_JSON)]}]
        )
        result = await agent.recommend_transfer("any query")
        assert isinstance(result, TransferRecommendationResponse)

    @pytest.mark.asyncio
    async def test_invalid_json_triggers_retry(self) -> None:
        from langchain_core.messages import AIMessage

        agent, mock_graph = _build_agent(
            [
                {"messages": [AIMessage(content=_INVALID_JSON)]},
                {"messages": [AIMessage(content=_INVALID_JSON)]},
                {"messages": [AIMessage(content=_VALID_RESPONSE_JSON)]},
            ]
        )
        result = await agent.recommend_transfer("query")
        assert isinstance(result, TransferRecommendationResponse)
        assert mock_graph.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_two_retries_exhausted_raises_llm_parsing_error(self) -> None:
        from langchain_core.messages import AIMessage

        agent, mock_graph = _build_agent(
            [
                {"messages": [AIMessage(content=_INVALID_JSON)]},
                {"messages": [AIMessage(content=_INVALID_JSON)]},
                {"messages": [AIMessage(content=_INVALID_JSON)]},
            ]
        )
        with pytest.raises(LLMParsingError):
            await agent.recommend_transfer("query")
        assert mock_graph.ainvoke.call_count == 3

    @pytest.mark.asyncio
    async def test_corrective_prompt_used_on_retry(self) -> None:
        from langchain_core.messages import AIMessage, HumanMessage

        agent, mock_graph = _build_agent(
            [
                {"messages": [AIMessage(content=_INVALID_JSON)]},
                {"messages": [AIMessage(content=_VALID_RESPONSE_JSON)]},
            ]
        )
        await agent.recommend_transfer("original query")
        second_call_messages = mock_graph.ainvoke.call_args_list[1][0][0]["messages"]
        assert isinstance(second_call_messages[0], HumanMessage)
        assert "original query" in second_call_messages[0].content
        assert "JSON" in second_call_messages[0].content


class TestBuildLlm:
    def test_openai_provider_returns_chat_openai(self) -> None:
        from orcanet.reasoning.agent import OrcaNetAgent

        mock_graph = MagicMock()
        with (
            patch("orcanet.reasoning.agent.create_agent", return_value=mock_graph),
            patch("langchain_openai.ChatOpenAI") as mock_cls,
        ):
            mock_cls.return_value = MagicMock()
            mock_cls.return_value.bind_tools = MagicMock(return_value=mock_cls.return_value)
            OrcaNetAgent(llm_provider="openai", api_key="k")
            mock_cls.assert_called_once()

    def test_anthropic_provider_returns_chat_anthropic(self) -> None:
        from orcanet.reasoning.agent import OrcaNetAgent

        mock_graph = MagicMock()
        with (
            patch("orcanet.reasoning.agent.create_agent", return_value=mock_graph),
            patch("langchain_anthropic.ChatAnthropic") as mock_cls,
        ):
            mock_cls.return_value = MagicMock()
            mock_cls.return_value.bind_tools = MagicMock(return_value=mock_cls.return_value)
            OrcaNetAgent(llm_provider="anthropic", api_key="k")
            mock_cls.assert_called_once()

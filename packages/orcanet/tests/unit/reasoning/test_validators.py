"""Unit tests for orcanet.reasoning.validators."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from orcanet.reasoning.validators import (
    LLMParsingError,
    SourceTaskRecommendation,
    TransferRecommendationResponse,
)

_VALID_SOURCE = {
    "task_id": "11111111-1111-1111-1111-111111111111",
    "task_name": "brain MRI classification",
    "similarity_score": 0.87,
    "transfer_score": 0.72,
    "reasoning": "Shared convolutional feature hierarchy",
}

_VALID_RESPONSE = {
    "top_sources": [_VALID_SOURCE],
    "recommended_strategy": "feature",
    "expected_improvement": 0.15,
    "explanation": "Domain similarity justifies feature-level transfer.",
    "confidence": 0.8,
}


class TestLLMParsingError:
    def test_is_exception_subclass(self) -> None:
        assert issubclass(LLMParsingError, Exception)

    def test_carries_message(self) -> None:
        err = LLMParsingError("parse failed after 3 attempts")
        assert "parse failed" in str(err)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(LLMParsingError, match="bad output"):
            raise LLMParsingError("bad output")


class TestSourceTaskRecommendation:
    def test_valid_construction(self) -> None:
        rec = SourceTaskRecommendation(**_VALID_SOURCE)
        assert rec.task_id == _VALID_SOURCE["task_id"]
        assert rec.task_name == _VALID_SOURCE["task_name"]
        assert rec.similarity_score == pytest.approx(0.87)
        assert rec.transfer_score == pytest.approx(0.72)
        assert rec.reasoning == _VALID_SOURCE["reasoning"]

    def test_similarity_score_below_zero_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SourceTaskRecommendation(**{**_VALID_SOURCE, "similarity_score": -0.1})

    def test_transfer_score_above_one_rejected(self) -> None:
        with pytest.raises(ValidationError):
            SourceTaskRecommendation(**{**_VALID_SOURCE, "transfer_score": 1.5})

    def test_missing_required_field_raises(self) -> None:
        incomplete = {k: v for k, v in _VALID_SOURCE.items() if k != "task_name"}
        with pytest.raises(ValidationError):
            SourceTaskRecommendation(**incomplete)


class TestTransferRecommendationResponse:
    def test_valid_parse_from_json(self) -> None:
        raw = json.dumps(_VALID_RESPONSE)
        resp = TransferRecommendationResponse.model_validate_json(raw)
        assert len(resp.top_sources) == 1
        assert resp.recommended_strategy == "feature"
        assert resp.expected_improvement == pytest.approx(0.15)
        assert resp.confidence == pytest.approx(0.8)

    def test_invalid_json_raises_validation_error(self) -> None:
        with pytest.raises((ValidationError, ValueError)):
            TransferRecommendationResponse.model_validate_json("{not valid json}")

    def test_confidence_above_one_rejected(self) -> None:
        bad = {**_VALID_RESPONSE, "confidence": 1.1}
        with pytest.raises(ValidationError):
            TransferRecommendationResponse(**bad)

    def test_expected_improvement_below_zero_rejected(self) -> None:
        bad = {**_VALID_RESPONSE, "expected_improvement": -0.05}
        with pytest.raises(ValidationError):
            TransferRecommendationResponse(**bad)

    def test_accepts_all_valid_strategy_names(self) -> None:
        for strategy in ("feature", "weight", "architecture", "multi_task"):
            resp = TransferRecommendationResponse(
                **{**_VALID_RESPONSE, "recommended_strategy": strategy}
            )
            assert resp.recommended_strategy == strategy

    def test_invalid_strategy_name_rejected(self) -> None:
        with pytest.raises(ValidationError):
            TransferRecommendationResponse(**{**_VALID_RESPONSE, "recommended_strategy": "invalid"})

    def test_empty_top_sources_is_valid(self) -> None:
        resp = TransferRecommendationResponse(**{**_VALID_RESPONSE, "top_sources": []})
        assert resp.top_sources == []

    def test_multiple_sources_parsed(self) -> None:
        response_with_two = {
            **_VALID_RESPONSE,
            "top_sources": [_VALID_SOURCE, {**_VALID_SOURCE, "task_id": "22222222-2222-2222-2222-222222222222"}],
        }
        resp = TransferRecommendationResponse(**response_with_two)
        assert len(resp.top_sources) == 2

    def test_boundary_values_accepted(self) -> None:
        resp = TransferRecommendationResponse(
            **{**_VALID_RESPONSE, "confidence": 0.0, "expected_improvement": 1.0}
        )
        assert resp.confidence == pytest.approx(0.0)
        assert resp.expected_improvement == pytest.approx(1.0)

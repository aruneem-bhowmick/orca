"""Unit tests for the recommendation explorer page – pure data functions."""

from __future__ import annotations

import importlib
import io
import sys
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(scope="module")
def re_mod(_patch_streamlit):
    sys.modules.pop("orcamind.dashboard.pages.recommendation_explorer", None)
    mod = importlib.import_module("orcamind.dashboard.pages.recommendation_explorer")
    return mod


def _make_csv(content: str) -> io.BytesIO:
    buf = io.BytesIO(content.encode())
    buf.name = "test.csv"
    return buf


# ── parse_csv_dataset ─────────────────────────────────────────────────────────

class TestParseCsvDataset:
    def test_returns_expected_shape(self, re_mod):
        csv = _make_csv("a,b,c,label\n1,2,3,cat\n4,5,6,dog\n7,8,9,cat\n")
        result = re_mod.parse_csv_dataset(csv)
        assert result["n_samples"] == 3
        assert result["n_features"] == 3  # 4 cols - 1 label
        assert result["n_classes"] == 2

    def test_task_type_is_classification(self, re_mod):
        csv = _make_csv("x,y\n1,0\n2,1\n3,0\n")
        result = re_mod.parse_csv_dataset(csv)
        assert result["task_type"] == "classification"

    def test_name_from_file_attribute(self, re_mod):
        csv = _make_csv("x,y\n1,0\n2,1\n")
        csv.name = "my_dataset.csv"
        result = re_mod.parse_csv_dataset(csv)
        assert result["name"] == "my_dataset.csv"

    def test_single_feature_column(self, re_mod):
        csv = _make_csv("feat,label\n1,a\n2,b\n3,a\n")
        result = re_mod.parse_csv_dataset(csv)
        assert result["n_features"] >= 1


# ── call_embed ────────────────────────────────────────────────────────────────

class TestCallEmbed:
    def test_returns_vector_from_embedding_vector_key(self, re_mod):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding_vector": [0.1, 0.2, 0.3]}
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp):
            result = re_mod.call_embed("http://localhost:8000", {"n_samples": 100})
        assert result == [0.1, 0.2, 0.3]

    def test_returns_vector_from_embedding_key(self, re_mod):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"embedding": [0.4, 0.5]}
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp):
            result = re_mod.call_embed("http://localhost:8000", {})
        assert result == [0.4, 0.5]

    def test_returns_list_directly(self, re_mod):
        mock_resp = MagicMock()
        mock_resp.json.return_value = [0.7, 0.8, 0.9]
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp):
            result = re_mod.call_embed("http://localhost:8000", {})
        assert result == [0.7, 0.8, 0.9]

    def test_raises_on_http_error(self, re_mod):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("500")
        with patch("requests.post", return_value=mock_resp):
            with pytest.raises(Exception, match="500"):
                re_mod.call_embed("http://localhost:8000", {})


# ── call_recommend ────────────────────────────────────────────────────────────

class TestCallRecommend:
    def test_returns_list_of_recommendations(self, re_mod):
        recs = [{"model_id": "m1", "predicted_score": 0.9}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = recs
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp):
            result = re_mod.call_recommend("http://localhost:8000", [0.1, 0.2], top_k=3)
        assert result == recs

    def test_passes_top_k_in_payload(self, re_mod):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp) as mock_post:
            re_mod.call_recommend("http://localhost:8000", [0.1], top_k=5)
        payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args.args[1]
        assert payload["top_k"] == 5


# ── call_similar_tasks ────────────────────────────────────────────────────────

class TestCallSimilarTasks:
    def test_returns_similar_task_list(self, re_mod):
        similar = [{"task_id": "t1", "similarity_score": 0.95}]
        mock_resp = MagicMock()
        mock_resp.json.return_value = similar
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp):
            result = re_mod.call_similar_tasks("http://localhost:8000", [0.1, 0.2])
        assert result == similar

    def test_url_targets_similar_tasks_endpoint(self, re_mod):
        mock_resp = MagicMock()
        mock_resp.json.return_value = []
        mock_resp.raise_for_status.return_value = None
        with patch("requests.post", return_value=mock_resp) as mock_post:
            re_mod.call_similar_tasks("http://localhost:8000", [0.1])
        url = mock_post.call_args.args[0]
        assert "similar-tasks" in url

"""Tests that verify the cross-domain transfer demo notebook is well-formed.

These tests do not execute the notebook cells (that would require a live Orca
stack).  Instead they validate the JSON structure, ensure all stub TODO markers
have been replaced with real code, and confirm each code cell contains
substantive content.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Fixture: load the notebook
# ---------------------------------------------------------------------------

_here = Path(__file__).resolve()
_pkg_root = _here.parent
while not (_pkg_root / "pyproject.toml").exists() and _pkg_root != _pkg_root.parent:
    _pkg_root = _pkg_root.parent

_NOTEBOOK_PATH = _pkg_root / "notebooks" / "cross_domain_transfer_demo.ipynb"


@pytest.fixture(scope="module")
def notebook() -> dict:
    """Load and return the parsed demo notebook JSON."""
    assert _NOTEBOOK_PATH.exists(), f"Notebook not found at {_NOTEBOOK_PATH}"
    with _NOTEBOOK_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def code_cells(notebook: dict) -> list[dict]:
    """Return only the code cells from the notebook."""
    return [c for c in notebook.get("cells", []) if c.get("cell_type") == "code"]


@pytest.fixture(scope="module")
def markdown_cells(notebook: dict) -> list[dict]:
    """Return only the markdown cells from the notebook."""
    return [c for c in notebook.get("cells", []) if c.get("cell_type") == "markdown"]


# ---------------------------------------------------------------------------
# Structure tests
# ---------------------------------------------------------------------------


class TestNotebookStructure:
    """Verify the notebook JSON structure is valid and complete."""

    def test_notebook_parses(self, notebook: dict) -> None:
        """The notebook file is valid JSON and has the expected top-level keys."""
        assert "cells" in notebook
        assert "metadata" in notebook
        assert notebook.get("nbformat") == 4

    def test_has_at_least_eight_code_cells(self, code_cells: list[dict]) -> None:
        """The demo notebook has at least one code cell per section (8 sections)."""
        assert len(code_cells) >= 8, (
            f"Expected ≥8 code cells, found {len(code_cells)}"
        )

    def test_has_markdown_section_headers(self, markdown_cells: list[dict]) -> None:
        """Each of the eight main sections has a markdown header."""
        headers = set()
        for cell in markdown_cells:
            source = "".join(cell.get("source", []))
            for i in range(1, 9):
                if f"## {i}." in source or f"# {i}." in source:
                    headers.add(i)
        assert len(headers) == 8, (
            f"Expected headers for sections 1–8, found headers for: {sorted(headers)}"
        )

    def test_first_cell_is_title_markdown(self, notebook: dict) -> None:
        """The first cell is a markdown title cell."""
        cells = notebook.get("cells", [])
        assert cells, "Notebook has no cells"
        first = cells[0]
        assert first.get("cell_type") == "markdown"
        source = "".join(first.get("source", []))
        assert source.strip().startswith("#"), "First cell is not a top-level heading"


# ---------------------------------------------------------------------------
# Content tests — no stub TODO placeholders
# ---------------------------------------------------------------------------


class TestNotebookContent:
    """Verify that notebook cells contain real code, not placeholder stubs."""

    def test_no_stub_print_statements(self, code_cells: list[dict]) -> None:
        """No code cell consists solely of a stub ``print('[stub] …')`` line."""
        stub_cells = []
        for i, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            lines = [line.strip() for line in source.splitlines() if line.strip()]
            if lines and all(line.startswith("print(\"[stub]") or line.startswith("print('[stub]") for line in lines):
                stub_cells.append(i)
        assert not stub_cells, (
            f"Code cells at indices {stub_cells} still contain only stub print statements"
        )

    def test_no_todo_implement_comments(self, code_cells: list[dict]) -> None:
        """No code cell contains a ``# TODO: implement`` placeholder comment."""
        offending = []
        for i, cell in enumerate(code_cells):
            source = "".join(cell.get("source", []))
            if "# TODO: implement" in source:
                offending.append(i)
        assert not offending, (
            f"Code cells at indices {offending} still have '# TODO: implement' placeholders"
        )

    def test_setup_cell_imports_httpx(self, code_cells: list[dict]) -> None:
        """The setup cell (first code cell) imports httpx."""
        setup_source = "".join(code_cells[0].get("source", []))
        assert "import httpx" in setup_source

    def test_setup_cell_defines_service_urls(self, code_cells: list[dict]) -> None:
        """The setup cell defines ORCAMIND_URL, ORCALAB_URL, and ORCANET_URL."""
        setup_source = "".join(code_cells[0].get("source", []))
        for var in ("ORCAMIND_URL", "ORCALAB_URL", "ORCANET_URL"):
            assert var in setup_source, f"Setup cell is missing {var!r} variable"

    def test_setup_cell_reads_env_vars(self, code_cells: list[dict]) -> None:
        """The setup cell reads service URLs from environment variables."""
        setup_source = "".join(code_cells[0].get("source", []))
        assert "os.getenv" in setup_source or "os.environ" in setup_source

    def test_embed_cell_uses_cross_domain_embedder(self, code_cells: list[dict]) -> None:
        """The embedding cell imports and uses CrossDomainEmbedder."""
        sources = ["".join(c.get("source", [])) for c in code_cells]
        combined = "\n".join(sources)
        assert "CrossDomainEmbedder" in combined

    def test_retrieve_cell_calls_retrieve_endpoint(self, code_cells: list[dict]) -> None:
        """At least one cell calls the /api/v1/retrieve endpoint."""
        sources = ["".join(c.get("source", [])) for c in code_cells]
        combined = "\n".join(sources)
        assert "/api/v1/retrieve" in combined

    def test_score_cell_calls_transfer_score_endpoint(self, code_cells: list[dict]) -> None:
        """At least one cell calls the /api/v1/transfer/score endpoint."""
        sources = ["".join(c.get("source", [])) for c in code_cells]
        combined = "\n".join(sources)
        assert "/api/v1/transfer/score" in combined

    def test_recommend_cell_calls_recommend_endpoint(self, code_cells: list[dict]) -> None:
        """At least one cell calls the /api/v1/transfer/recommend endpoint."""
        sources = ["".join(c.get("source", [])) for c in code_cells]
        combined = "\n".join(sources)
        assert "/api/v1/transfer/recommend" in combined

    def test_validate_cell_calls_validate_endpoint(self, code_cells: list[dict]) -> None:
        """At least one cell calls the /api/v1/transfer/validate endpoint."""
        sources = ["".join(c.get("source", [])) for c in code_cells]
        combined = "\n".join(sources)
        assert "/api/v1/transfer/validate" in combined

    def test_summary_cell_uses_matplotlib(self, code_cells: list[dict]) -> None:
        """The final summary cell produces a bar chart with matplotlib."""
        sources = ["".join(c.get("source", [])) for c in code_cells]
        combined = "\n".join(sources)
        assert "plt" in combined and ("bar" in combined or "barh" in combined)

    def test_summary_cell_has_demo_fallback(self, code_cells: list[dict]) -> None:
        """The summary section has a demo fallback table for offline use."""
        last_code = "".join(code_cells[-1].get("source", []))
        assert "Demo" in last_code or "demo" in last_code, (
            "Last code cell should include a demo fallback output for offline use"
        )

    def test_umap_import_is_guarded(self, code_cells: list[dict]) -> None:
        """UMAP import is wrapped in a try/except ImportError so the notebook degrades gracefully."""
        sources = ["".join(c.get("source", [])) for c in code_cells]
        combined = "\n".join(sources)
        assert "ImportError" in combined, (
            "UMAP import should be guarded with try/except ImportError"
        )

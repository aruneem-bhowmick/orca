"""LLM-powered query expander for broadening task-similarity search recall."""

from __future__ import annotations

import re

from langchain_core.language_models import BaseLLM


class QueryExpander:
    """Generates alternative phrasings of an ML task description using an LLM."""

    def __init__(self, llm: BaseLLM) -> None:
        self._llm = llm

    async def expand(self, query: str, n_expansions: int = 3) -> list[str]:
        """Return *n_expansions* alternative descriptions of *query*."""
        prompt = f"Generate {n_expansions} alternative descriptions for this ML task: {query}"
        response = await self._llm.ainvoke(prompt)
        text = response if isinstance(response, str) else getattr(response, "content", "")
        return _parse_list_from_response(text)


def _parse_list_from_response(text: str) -> list[str]:
    """Strip bullets/numbering from each line and return non-empty strings."""
    results = []
    for line in text.splitlines():
        line = re.sub(r"^\s*(?:\d+[.)]\s*|[-*•]\s*)", "", line).strip()
        if line:
            results.append(line)
    return results

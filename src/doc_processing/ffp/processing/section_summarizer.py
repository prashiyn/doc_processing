from __future__ import annotations

"""
Section-level metadata summaries for chunks.

Each chunk receives a short summary derived from its section text,
using remote llm-service via runtime client.
"""

from collections import defaultdict
from typing import Any, Iterable

from doc_processing.llm_runtime import HttpLLMRuntime


class SectionSummarizer:
    """Generate section-level summaries and attach them to chunks."""

    def __init__(self, client: HttpLLMRuntime | None = None) -> None:
        self._client = client or HttpLLMRuntime()

    def summarize(self, chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach `title_summary` to each chunk based on its section's text."""
        chunks_list = list(chunks)
        if not chunks_list:
            return []

        # Collect concatenated text per section_title.
        per_section: dict[str | None, list[str]] = defaultdict(list)
        for c in chunks_list:
            section_title = c.get("section_title")
            content = c.get("content") or ""
            per_section[section_title].append(content)

        section_summaries: dict[str | None, str] = {}
        for section_title, texts in per_section.items():
            section_text = "\n\n".join(texts).strip()
            if not section_text:
                section_summaries[section_title] = ""
                continue

            title = section_title or "This section"
            prompt = (
                "Summarize the following section from a financial filing in one concise sentence. "
                "Focus on the main financial insight or key point.\n\n"
                f"Section title: {title}\n\n"
                f"Section text:\n{section_text}"
            )
            messages = [{"role": "user", "content": prompt}]
            summary = self._client.complete_with_fallback(
                messages,
                use_case="chunk_section_summary",
            )
            section_summaries[section_title] = (summary or "").strip()

        out: list[dict[str, Any]] = []
        for c in chunks_list:
            new_c = dict(c)
            new_c["title_summary"] = section_summaries.get(c.get("section_title"), "")
            out.append(new_c)

        return out


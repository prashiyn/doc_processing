from __future__ import annotations

"""
Section-level metadata summaries for chunks.

Each chunk receives a short summary derived from its section text,
using an LLM model configured via `config/chunking.yaml`
(`models.section_summarization_model`).
"""

from collections import defaultdict
from typing import Any, Iterable

import yaml

from doc_processing.config import get_config_dir
from doc_processing.llms.client import LLMClient


def _load_summary_model() -> str | None:
    path = get_config_dir() / "chunking.yaml"
    if not path.exists():
        return None
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    models = cfg.get("models") or {}
    model = models.get("section_summarization_model")
    return str(model) if model else None


class SectionSummarizer:
    """Generate section-level summaries and attach them to chunks."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or LLMClient()
        self._model = _load_summary_model()

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
            summary = self._client.complete_with_fallback(messages, model=self._model)
            section_summaries[section_title] = (summary or "").strip()

        out: list[dict[str, Any]] = []
        for c in chunks_list:
            new_c = dict(c)
            new_c["title_summary"] = section_summaries.get(c.get("section_title"), "")
            out.append(new_c)

        return out


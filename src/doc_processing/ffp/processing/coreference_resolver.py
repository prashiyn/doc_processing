from __future__ import annotations

"""
Coreference resolution over chunks using an LLM.

Uses a sliding window of previous chunks as context (same section title),
with window size configured via `config/chunking.yaml` (params.coreference_context_k).
LLM calls are routed via remote llm-service runtime.
"""

from typing import Any, Iterable

import yaml

from doc_processing.config import get_config_dir
from doc_processing.llm_runtime import HttpLLMRuntime


def _load_coref_config() -> int:
    path = get_config_dir() / "chunking.yaml"
    if not path.exists():
        return 4
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    params = cfg.get("params") or {}
    try:
        k = int(params.get("coreference_context_k", 4))
    except (TypeError, ValueError):
        k = 4
    return k


class CoreferenceResolver:
    """LLM-based coreference resolver for chunk text."""

    def __init__(self, client: HttpLLMRuntime | None = None) -> None:
        self._client = client or HttpLLMRuntime()
        self._k = _load_coref_config()

    def resolve(self, chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Rewrite each chunk's text to resolve pronouns using nearby context."""
        resolved: list[dict[str, Any]] = []
        history: list[dict[str, Any]] = []

        for chunk in chunks:
            text = chunk.get("content") or ""
            section_title = chunk.get("section_title")

            # build context from previous k chunks with same section_title
            ctx_pieces: list[str] = []
            for prev in reversed(history):
                if len(ctx_pieces) >= self._k:
                    break
                if prev.get("section_title") != section_title:
                    continue
                ctx_pieces.append(prev.get("content") or "")
            ctx = "\n\n".join(reversed(ctx_pieces)).strip()

            if not text.strip():
                resolved.append(chunk)
                history.append(chunk)
                continue

            prompt = (
                "Resolve pronouns in the following text by replacing them with the most "
                "likely explicit entities, using the provided context from the same section.\n\n"
                f"Context:\n{ctx or '[no prior context]'}\n\n"
                f"Text:\n{text}\n\n"
                "Rewrite the text so that pronouns like 'it', 'they', 'he', 'she', 'this', "
                "and 'that' are replaced with concrete entities when possible. "
                "Preserve the meaning and keep the same language."
            )
            messages = [{"role": "user", "content": prompt}]
            new_text = self._client.complete_with_fallback(
                messages,
                use_case="chunk_coreference",
            ) or text
            new_chunk = dict(chunk)
            new_chunk["content"] = new_text.strip() or text
            resolved.append(new_chunk)
            history.append(new_chunk)

        return resolved


from __future__ import annotations

"""
Cross-reference extraction from financial filing chunk text (LLM + post-processing).

Prompts and reference taxonomy follow `docs/REFERENCES.md`.
Model/provider selection: `config/llm_config.yaml` use-case mapping.
"""

import json
import re
from typing import Any, Iterable

from doc_processing.llm_runtime import HttpLLMRuntime

REFERENCE_SYSTEM_PROMPT = """You are a financial document structure analysis engine.
Extract all cross-references in the text. They may point to:
tables, figures, sections, subsections, appendices, annexures, notes, footnotes,
schedules, financial statements, or regulatory standards (e.g. Ind AS, Companies Act).

Ignore vague phrases like "as mentioned above" or "as discussed earlier".
Do not hallucinate; only extract references explicitly present.

Return strict JSON only with this shape:
{"references":[{"reference_text":"...","reference_type":"TABLE","target_label":"...","confidence":0.9}]}

reference_type must be one of:
TABLE, FIGURE, SECTION, SUBSECTION, APPENDIX, NOTE, FOOTNOTE, SCHEDULE, STATEMENT, REGULATION, OTHER

If none, return {"references":[]}.

Examples:
Text: Revenue (refer Table 3) in Note 12.
{"references":[{"reference_text":"Table 3","reference_type":"TABLE","target_label":"3","confidence":0.95},{"reference_text":"Note 12","reference_type":"NOTE","target_label":"12","confidence":0.95}]}

Text: per Ind AS 115 and Schedule III.
{"references":[{"reference_text":"Ind AS 115","reference_type":"REGULATION","target_label":"Ind AS 115","confidence":0.98},{"reference_text":"Schedule III","reference_type":"SCHEDULE","target_label":"III","confidence":0.95}]}

Text: See Appendix A and footnote 4.
{"references":[{"reference_text":"Appendix A","reference_type":"APPENDIX","target_label":"A","confidence":0.95},{"reference_text":"footnote 4","reference_type":"FOOTNOTE","target_label":"4","confidence":0.9}]}
"""

ALLOWED_TYPES = frozenset(
    {
        "TABLE",
        "FIGURE",
        "SECTION",
        "SUBSECTION",
        "APPENDIX",
        "NOTE",
        "FOOTNOTE",
        "SCHEDULE",
        "STATEMENT",
        "REGULATION",
        "OTHER",
    }
)

CONFIDENCE_MIN = 0.7


def _strip_json_fence(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
        s = re.sub(r"\s*```\s*$", "", s)
    return s.strip()


def _coerce_float(v: Any, default: float = 0.0) -> float:
    try:
        x = float(v)
        if x < 0.0:
            return 0.0
        if x > 1.0:
            return 1.0
        return x
    except (TypeError, ValueError):
        return default


def post_process_references(raw: list[Any]) -> list[dict[str, Any]]:
    """
    Validate, normalize labels, deduplicate by reference_text, apply confidence floor.
    """
    out: list[dict[str, Any]] = []
    seen_text: set[str] = set()

    for item in raw:
        if not isinstance(item, dict):
            continue
        text = item.get("reference_text")
        if not isinstance(text, str) or not text.strip():
            continue
        text = text.strip()
        key = text.casefold()
        if key in seen_text:
            continue

        rtype = item.get("reference_type")
        if not isinstance(rtype, str) or not rtype.strip():
            rtype = "OTHER"
        else:
            rtype = rtype.strip().upper()
            if rtype not in ALLOWED_TYPES:
                rtype = "OTHER"

        label = item.get("target_label")
        if isinstance(label, str):
            label = label.strip()
        elif label is not None:
            label = str(label).strip()
        else:
            label = ""

        conf = _coerce_float(item.get("confidence"), 0.0)
        # docs/REFERENCES.md: keep references with confidence > 0.7
        if conf <= CONFIDENCE_MIN:
            continue

        seen_text.add(key)
        out.append(
            {
                "reference_text": text,
                "reference_type": rtype,
                "target_label": label,
                "confidence": round(conf, 4),
            }
        )

    return out


class ReferenceExtractor:
    """Extract structured cross-references from each chunk's text content."""

    def __init__(self, client: HttpLLMRuntime | None = None) -> None:
        self._client = client or HttpLLMRuntime()

    def extract_for_chunk(self, content: str) -> list[dict[str, Any]]:
        text = (content or "").strip()
        if not text:
            return []

        user_prompt = (
            "Extract all cross-references from the following financial text.\n\n"
            f"TEXT:\n{text}"
        )
        messages = [
            {"role": "system", "content": REFERENCE_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        try:
            raw_json = self._client.complete_with_fallback(
                messages,
                use_case="chunk_reference_extraction",
                response_format={"type": "json_object"},
            )
        except Exception:
            raw_json = self._client.complete_with_fallback(
                messages,
                use_case="chunk_reference_extraction",
            )

        raw_json = _strip_json_fence(raw_json or "")
        if not raw_json:
            return []

        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            return []

        if not isinstance(data, dict):
            return []

        refs = data.get("references")
        if not isinstance(refs, list):
            return []

        return post_process_references(refs)

    def enrich_chunks(self, chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Attach `references` to each chunk dict (copy)."""
        result: list[dict[str, Any]] = []
        for c in chunks:
            nc = dict(c)
            content = nc.get("content") or ""
            nc["references"] = self.extract_for_chunk(str(content))
            result.append(nc)
        return result

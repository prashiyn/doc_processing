from __future__ import annotations

"""
Table-to-text conversion using the shared LLM client.

Model selection is controlled by `config/llm_config.yaml` use-case mapping.
"""

from pathlib import Path

import base64

from doc_processing.llm_runtime import HttpLLMRuntime


class TableExtractor:
    """Convert table images into concise textual explanations."""

    def __init__(self, client: HttpLLMRuntime | None = None) -> None:
        self._client = client or HttpLLMRuntime()

    @staticmethod
    def _encode_image(path: str | Path) -> str:
        p = Path(path)
        data = p.read_bytes()
        return base64.b64encode(data).decode("ascii")

    def convert(self, img_path: str | Path) -> str:
        """Convert a table image into a short textual summary."""
        image_b64 = self._encode_image(img_path)

        system_prompt = (
            "You are a financial analysis assistant. "
            "You are given an image of a financial table. "
            "Convert this financial table into a concise textual explanation."
            "Describe the key metrics, notable trends, and any obvious relationships. "
            "Be factual and concise. Do not hallucinate values that are not visible."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Here is the table image (base64-encoded): {image_b64}",
            },
        ]

        return self._client.complete_with_fallback(
            messages,
            use_case="chunk_table_extraction",
        )


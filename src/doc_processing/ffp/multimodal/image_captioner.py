from __future__ import annotations

"""
Figure/chart captioning using the shared LLM client.

Model selection is controlled by `config/llm_config.yaml` use-case mapping.
"""

from pathlib import Path

import base64

from doc_processing.llm_runtime import HttpLLMRuntime


class ImageCaptioner:
    """Caption financial figures (charts, diagrams) with key insights."""

    def __init__(self, client: HttpLLMRuntime | None = None) -> None:
        self._client = client or HttpLLMRuntime()

    @staticmethod
    def _encode_image(path: str | Path) -> str:
        p_str = str(path)
        if p_str.startswith("http://") or p_str.startswith("https://"):
            import requests

            r = requests.get(p_str, timeout=30)
            r.raise_for_status()
            return base64.b64encode(r.content).decode("ascii")
        p = Path(path)
        data = p.read_bytes()
        return base64.b64encode(data).decode("ascii")

    def caption(self, img_path: str | Path) -> str:
        """Generate a caption describing the main financial insight of a figure."""
        image_b64 = self._encode_image(img_path)

        system_prompt = (
            "You are a financial analysis assistant. "
            "You are given an image of a chart or figure from a financial filing. "
            "Describe the key financial insight, focusing on trends or important metrics. "
            "Be concise (1–3 sentences) and do not invent data."
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Here is the figure image (base64-encoded): {image_b64}",
            },
        ]

        return self._client.complete_with_fallback(
            messages,
            use_case="chunk_image_caption",
        )


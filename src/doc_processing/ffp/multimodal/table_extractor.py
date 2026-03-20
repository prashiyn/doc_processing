from __future__ import annotations

"""
Table-to-text conversion using the shared LLM client.

Model selection is controlled by `config/chunking.yaml` (models.table_extraction_model).
"""

from pathlib import Path
from typing import Any

import base64
import yaml

from doc_processing.config import get_config_dir
from doc_processing.llms.client import LLMClient


def _load_chunking_config() -> dict[str, Any]:
    path = get_config_dir() / "chunking.yaml"
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def _get_table_model_name(cfg: dict[str, Any]) -> str | None:
    models = cfg.get("models") or {}
    model = models.get("table_extraction_model")
    return str(model) if model else None


class TableExtractor:
    """Convert table images into concise textual explanations."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self._client = client or LLMClient()
        self._cfg = _load_chunking_config()
        self._model = _get_table_model_name(self._cfg)

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

        return self._client.complete_with_fallback(messages, model=self._model)


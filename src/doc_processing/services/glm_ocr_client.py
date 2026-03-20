"""
Minimal GLM-OCR client that calls Ollama /api/generate for document OCR.

Implements the same contract as the upstream GlmOcr.parse() → markdown result,
without depending on the glmocr package (which conflicts with docling).
See: https://github.com/zai-org/GLM-OCR (api.py, config, pipeline_result).
"""

from __future__ import annotations

import base64
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)

# Default prompt from GLM-OCR PageLoaderConfig (config.py)
_DEFAULT_PROMPT = (
    "Recognize the text in the image and output in Markdown format. "
    "Preserve the original layout (headings/paragraphs/tables/formulas). "
    "Do not fabricate content that does not exist in the image."
)


@dataclass
class GlmOcrConfig:
    """Ollama OCR API settings (mirrors pipeline.ocr_api in GLM-OCR config)."""
    api_host: str = "localhost"
    api_port: int = 11434
    api_path: str = "/api/generate"
    model: str = "glm-ocr:latest"
    request_timeout: int = 300


def load_config(config_path: str | Path | None) -> GlmOcrConfig:
    """Load config from YAML (project config/glm_ocr_ollama.yaml structure) or defaults."""
    cfg = GlmOcrConfig()
    if not config_path:
        return cfg
    path = Path(config_path)
    if not path.exists():
        return cfg
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        ocr = (data.get("pipeline") or {}).get("ocr_api") or {}
        if ocr.get("api_host") is not None:
            cfg.api_host = str(ocr["api_host"])
        if ocr.get("api_port") is not None:
            cfg.api_port = int(ocr["api_port"])
        if ocr.get("api_path") is not None:
            cfg.api_path = str(ocr["api_path"]).strip() or "/api/generate"
        if ocr.get("model") is not None:
            cfg.model = str(ocr["model"])
        if ocr.get("request_timeout") is not None:
            cfg.request_timeout = int(ocr["request_timeout"])
    except Exception:
        pass
    return cfg


@dataclass
class PipelineResult:
    """Minimal result: markdown and optional json/original_images for compatibility."""
    markdown_result: str
    json_result: list[Any]
    original_images: list[str]

    def __post_init__(self) -> None:
        if self.json_result is None:
            self.json_result = []
        if self.original_images is None:
            self.original_images = []


class GlmOcrClient:
    """
    Minimal GLM-OCR client using Ollama /api/generate.

    Same usage as upstream: GlmOcr(config_path="...") then parse(path) or parse([paths]).
    Returns PipelineResult (with markdown_result) or list of PipelineResult for list input.
    """

    def __init__(
        self,
        config_path: str | Path | None = None,
        *,
        api_host: str | None = None,
        api_port: int | None = None,
        api_path: str | None = None,
        model: str | None = None,
        request_timeout: int | None = None,
    ) -> None:
        self._config = load_config(config_path)
        if api_host is not None:
            self._config.api_host = api_host
        if api_port is not None:
            self._config.api_port = api_port
        if api_path is not None:
            self._config.api_path = api_path
        if model is not None:
            self._config.model = model
        if request_timeout is not None:
            self._config.request_timeout = request_timeout

    @property
    def _base_url(self) -> str:
        return f"http://{self._config.api_host}:{self._config.api_port}{self._config.api_path}"

    @property
    def _is_chat_api(self) -> bool:
        return "/chat" in (self._config.api_path or "").rstrip("/").lower()

    def _read_image_as_base64(self, path: str | Path) -> str:
        """Read image file and return base64-encoded string (no data URI prefix)."""
        raw = Path(path).read_bytes()
        return base64.b64encode(raw).decode("ascii")

    def _request_one(self, b64_images: list[str], prompt: str) -> str:
        """Send one request (generate or chat) and return the extracted text. Logs on failure."""
        if self._is_chat_api:
            payload: dict[str, Any] = {
                "model": self._config.model,
                "messages": [
                    {"role": "user", "content": prompt, "images": b64_images}
                ],
                "stream": False,
            }
        else:
            payload = {
                "model": self._config.model,
                "prompt": prompt,
                "stream": False,
                "images": b64_images,
            }
        try:
            r = requests.post(
                self._base_url,
                json=payload,
                timeout=self._config.request_timeout,
            )
            r.raise_for_status()
            data = r.json()
            if self._is_chat_api:
                msg = data.get("message") or {}
                text = (msg.get("content") or "").strip()
            else:
                text = (data.get("response") or "").strip()
            if not text:
                logger.warning(
                    "Ollama returned 200 but response was empty (model=%s, path=%s, images=%s). "
                    "Check model name (e.g. ollama list) and that the model supports vision.",
                    self._config.model,
                    self._config.api_path,
                    len(b64_images),
                )
            return text
        except requests.exceptions.RequestException as e:
            logger.warning(
                "GLM-OCR request failed: %s (url=%s, model=%s)",
                e,
                self._base_url,
                self._config.model,
            )
            return ""
        except Exception as e:
            logger.warning("GLM-OCR request error: %s", e, exc_info=True)
            return ""

    def parse(
        self,
        images: str | list[str],
        prompt: str = _DEFAULT_PROMPT,
    ) -> PipelineResult | list[PipelineResult]:
        """
        Run OCR on one or more images (file paths). Uses Ollama /api/generate or /api/chat.

        - Single path (str) → one request, returns one PipelineResult.
        - List of paths (e.g. PDF pages) → one request per page, markdown concatenated into
          one PipelineResult (avoids timeouts and yields partial output per page).
        """
        single = isinstance(images, str)
        paths = [images] if single else list(images)
        if not paths:
            return PipelineResult("", [], []) if single else []

        image_paths = [str(Path(p).resolve()) for p in paths]
        b64_list = [self._read_image_as_base64(p) for p in paths]

        # Multi-page: one request per image to avoid timeouts and get per-page output
        if len(b64_list) > 1:
            parts: list[str] = []
            for i, b64 in enumerate(b64_list):
                page_text = self._request_one([b64], prompt)
                if page_text:
                    parts.append(page_text)
                else:
                    logger.warning("Page %s returned empty OCR result.", i + 1)
            text = "\n\n".join(parts)
        else:
            text = self._request_one(b64_list, prompt)

        result = PipelineResult(
            markdown_result=text,
            json_result=[],
            original_images=image_paths,
        )
        return result if single else [result]

    def close(self) -> None:
        """No-op for compatibility with upstream context manager."""
        pass

    def __enter__(self) -> GlmOcrClient:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

"""
Embedding wrapper for LiteLLM with direct Ollama fallback.

For most providers we call LiteLLM's `embedding()` / `aembedding()`.
For `ollama/...` models, we call Ollama directly to avoid provider gaps.
"""

from __future__ import annotations

from typing import Any

import requests

from doc_processing.config import get_settings


class EmbeddingClient:
    """Thin embedding client used by API routes and internal services."""

    def __init__(self, default_model: str | None = None):
        self._default_model = default_model or "text-embedding-3-small"

    async def aembed(
        self,
        input_data: str | list[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """
        Async embedding call.

        Returns an OpenAI-style response:
        {
          "object": "list",
          "model": "...",
          "data": [{"object":"embedding","index":0,"embedding":[...]}],
          "usage": {...}
        }
        """
        model_name = model or self._default_model
        if model_name.startswith("ollama/"):
            return await self._aembed_ollama(input_data, model_name)

        from litellm import aembedding

        resp = await aembedding(model=model_name, input=input_data, **kwargs)
        return self._to_dict(resp)

    def embed(
        self,
        input_data: str | list[str],
        *,
        model: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Sync embedding call with same behavior as `aembed`."""
        model_name = model or self._default_model
        if model_name.startswith("ollama/"):
            return self._embed_ollama(input_data, model_name)

        from litellm import embedding

        resp = embedding(model=model_name, input=input_data, **kwargs)
        return self._to_dict(resp)

    async def _aembed_ollama(self, input_data: str | list[str], model: str) -> dict[str, Any]:
        # Keep async interface while using requests for simple consistency.
        return self._embed_ollama(input_data, model)

    def _embed_ollama(self, input_data: str | list[str], model: str) -> dict[str, Any]:
        settings = get_settings()
        base_url = settings.ollama_base_url.rstrip("/")
        model_name = model.split("/", 1)[1] if "/" in model else model

        # Preferred modern Ollama endpoint supporting batch input.
        payload = {"model": model_name, "input": input_data}
        r = requests.post(f"{base_url}/api/embed", json=payload, timeout=120)
        if r.status_code == 404:
            # Backward compatibility for older Ollama releases.
            legacy_prompt = input_data[0] if isinstance(input_data, list) else input_data
            legacy = {"model": model_name, "prompt": legacy_prompt}
            r = requests.post(f"{base_url}/api/embeddings", json=legacy, timeout=120)
        r.raise_for_status()
        body = r.json()

        # Normalize Ollama responses into OpenAI-style embedding payload.
        if "embeddings" in body and isinstance(body["embeddings"], list):
            vectors = body["embeddings"]
        elif "embedding" in body:
            vectors = [body["embedding"]]
        else:
            raise ValueError("Unexpected Ollama embedding response: missing embedding vectors.")

        data = [
            {"object": "embedding", "index": i, "embedding": vec}
            for i, vec in enumerate(vectors)
        ]
        return {
            "object": "list",
            "model": model,
            "data": data,
            "usage": body.get("usage"),
        }

    @staticmethod
    def _to_dict(resp: Any) -> dict[str, Any]:
        if isinstance(resp, dict):
            return resp
        if hasattr(resp, "model_dump"):
            return resp.model_dump()
        if hasattr(resp, "dict"):
            return resp.dict()
        raise TypeError("Unsupported embedding response type")


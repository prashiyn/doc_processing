from __future__ import annotations

"""Remote runtime client for doc-processing -> llm-service API calls."""

from typing import Any

import requests

from doc_processing.config import get_settings
from doc_processing.llm_runtime.config import get_service_llm_runtime_config, get_use_case_llm_config


def _provider_from_model(model: str | None) -> str:
    if not model:
        return "openai"
    if "/" in model:
        return model.split("/", 1)[0]
    return "openai"


class HttpLLMRuntime:
    """
    Compatibility wrapper for LLM operations over standalone llm-service APIs.

    It supports use-case driven model/provider selection via `src/config/llm_config.yaml`,
    while allowing explicit per-call model overrides.
    """

    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout_seconds: float | None = None,
        service_auth_token: str | None = None,
    ) -> None:
        settings = get_settings()
        service_cfg = get_service_llm_runtime_config()

        cfg_timeout = service_cfg.get("timeout_seconds")

        resolved_url = base_url or settings.llm_service_base_url
        resolved_timeout = timeout_seconds
        if resolved_timeout is None:
            if isinstance(cfg_timeout, (int, float)):
                resolved_timeout = float(cfg_timeout)
            else:
                resolved_timeout = 120.0

        self._base_url = resolved_url.rstrip("/")
        self._timeout = float(resolved_timeout)
        self._token = service_auth_token if service_auth_token is not None else settings.service_auth_token

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["X-Service-Token"] = self._token
        return headers

    @staticmethod
    def _resolve_provider_model(use_case: str | None, model: str | None) -> tuple[str, str | None]:
        case_cfg = get_use_case_llm_config(use_case)
        resolved_model = model or (str(case_cfg.get("model")) if case_cfg.get("model") else None)
        if model:
            provider = _provider_from_model(resolved_model)
        else:
            provider = str(case_cfg.get("provider")) if case_cfg.get("provider") else _provider_from_model(resolved_model)
        return provider, resolved_model

    def complete_with_fallback(
        self,
        messages: list[dict[str, str]],
        *,
        use_case: str | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        provider, resolved_model = self._resolve_provider_model(use_case, model)
        payload: dict[str, Any] = {
            "provider": provider,
            "messages": messages,
            "model": resolved_model,
        }
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort
        if response_format is not None:
            payload["response_format"] = response_format

        try:
            r = requests.post(
                f"{self._base_url}/llm/complete",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"llm-service completion call failed: {e}") from e

        body = r.json()
        content = body.get("content")
        if not isinstance(content, str):
            raise RuntimeError("Invalid completion response from llm-service: missing string content")
        return content

    def embed(
        self,
        input_data: str | list[str],
        *,
        use_case: str | None = None,
        model: str | None = None,
        encoding_format: str | None = None,
        dimensions: int | None = None,
        input_type: str | None = None,
        user: str | None = None,
    ) -> dict[str, Any]:
        provider, resolved_model = self._resolve_provider_model(use_case, model)
        payload: dict[str, Any] = {
            "provider": provider,
            "input": input_data,
            "model": resolved_model,
        }
        if encoding_format is not None:
            payload["encoding_format"] = encoding_format
        if dimensions is not None:
            payload["dimensions"] = dimensions
        if input_type is not None:
            payload["input_type"] = input_type
        if user is not None:
            payload["user"] = user

        try:
            r = requests.post(
                f"{self._base_url}/llm/embeddings",
                json=payload,
                headers=self._headers(),
                timeout=self._timeout,
            )
            r.raise_for_status()
        except requests.RequestException as e:
            raise RuntimeError(f"llm-service embeddings call failed: {e}") from e

        body = r.json()
        if not isinstance(body, dict):
            raise RuntimeError("Invalid embeddings response from llm-service")
        return body

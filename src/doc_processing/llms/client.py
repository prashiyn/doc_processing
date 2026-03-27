"""
Wrapper around LiteLLM for completion and async completion.
Uses config/llms.yaml for default/fallback models; API keys from env (LiteLLM convention).
Groq models (model.startswith("groq/")) are rate-limited via llms.groq_ratelimit.
"""
from __future__ import annotations

from typing import Any

from doc_processing.llms.config import get_default_model, get_fallback_model


def _supports_response_format(model: str) -> bool:
    """
    Check whether the model/provider supports OpenAI-style `response_format`.
    See LiteLLM docs: Structured Outputs (JSON Mode).
    """
    try:
        import litellm

        params = litellm.get_supported_openai_params(model=model)
        return "response_format" in (params or [])
    except Exception:
        # Be conservative: if LiteLLM can't determine support, treat as unsupported.
        return False


def _supports_response_schema(model: str) -> bool:
    """
    Check whether the model/provider supports `response_format` as a JSON schema / Pydantic schema.
    See LiteLLM docs: supports_response_schema().
    """
    try:
        import litellm

        return bool(litellm.supports_response_schema(model=model))
    except Exception:
        return False


def _ensure_structured_output_supported(model: str, response_format: Any) -> None:
    """
    Validate that structured output parameters are likely supported for the model.
    Raises ValueError with a concise message if unsupported.
    """
    if response_format is None:
        return

    # LiteLLM accepts:
    # - {"type": "json_object"}
    # - {"type": "json_schema", "json_schema": {...}, "strict": True}
    # - A Pydantic model (python-only; not expected from HTTP routes)
    if isinstance(response_format, dict):
        rf_type = str(response_format.get("type", "")).strip().lower()
        if rf_type == "json_object":
            if not _supports_response_format(model):
                raise ValueError(f"Model does not support response_format=json_object: {model}")
            return
        if rf_type == "json_schema":
            if not _supports_response_schema(model):
                raise ValueError(f"Model does not support response_format=json_schema: {model}")
            return
        # Unknown dict format – let LiteLLM try, but provide a clearer error early.
        raise ValueError("Invalid response_format. Expected type=json_object or type=json_schema.")

    # Non-dict response_format (e.g. Pydantic model): require response_schema support.
    if not _supports_response_schema(model):
        raise ValueError(f"Model does not support response schema output: {model}")


def _supports_reasoning(model: str) -> bool:
    """Check whether the model/provider supports reasoning parameters."""
    try:
        import litellm

        return bool(litellm.supports_reasoning(model=model))
    except Exception:
        return False


def _ensure_reasoning_supported(model: str, reasoning_effort: Any) -> None:
    """Validate `reasoning_effort` usage for the selected model."""
    if reasoning_effort is None:
        return
    if reasoning_effort not in ("low", "medium", "high"):
        raise ValueError("Invalid reasoning_effort. Expected one of: low, medium, high.")
    if not _supports_reasoning(model):
        raise ValueError(f"Model does not support reasoning_effort: {model}")


def _apply_groq_rate_limit(model: str) -> None:
    """Block until a Groq request is allowed; no-op for non-Groq models."""
    from doc_processing.llms.groq_ratelimit import get_groq_rate_limiter
    get_groq_rate_limiter().wait_if_needed(model)


def _record_groq_request(model: str) -> None:
    """Record a Groq request for rate limiting; no-op for non-Groq models."""
    from doc_processing.llms.groq_ratelimit import get_groq_rate_limiter
    get_groq_rate_limiter().record_request(model)


class LLMClient:
    """
    Thin wrapper over LiteLLM for direct and agent use.
    Customize behavior here (retries, logging, model routing) without changing call sites.
    """

    def __init__(self, default_model: str | None = None, fallback_model: str | None = None):
        self._default_model = default_model or get_default_model()
        self._fallback_model = fallback_model or get_fallback_model()

    def complete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """
        Synchronous completion. Returns the assistant message content.
        Raises on error; use complete_with_fallback for automatic fallback model.
        Groq models are rate-limited (RPM/RPD) before each request.
        """
        import litellm
        model = model or self._default_model
        _ensure_reasoning_supported(model, kwargs.get("reasoning_effort"))
        _ensure_structured_output_supported(model, kwargs.get("response_format"))
        _apply_groq_rate_limit(model)
        response = litellm.completion(model=model, messages=messages, **kwargs)
        _record_groq_request(model)
        return self._extract_content(response)

    def complete_with_fallback(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Run completion; on failure try fallback model once. Groq models are rate-limited."""
        import litellm
        model = model or self._default_model
        try:
            _ensure_reasoning_supported(model, kwargs.get("reasoning_effort"))
            _ensure_structured_output_supported(model, kwargs.get("response_format"))
            _apply_groq_rate_limit(model)
            response = litellm.completion(model=model, messages=messages, **kwargs)
            _record_groq_request(model)
            return self._extract_content(response)
        except Exception:
            _ensure_reasoning_supported(self._fallback_model, kwargs.get("reasoning_effort"))
            _ensure_structured_output_supported(self._fallback_model, kwargs.get("response_format"))
            _apply_groq_rate_limit(self._fallback_model)
            response = litellm.completion(
                model=self._fallback_model, messages=messages, **kwargs
            )
            _record_groq_request(self._fallback_model)
            return self._extract_content(response)

    async def acomplete(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Async completion. Returns the assistant message content. Groq models are rate-limited."""
        from litellm import acompletion
        model = model or self._default_model
        _ensure_reasoning_supported(model, kwargs.get("reasoning_effort"))
        _ensure_structured_output_supported(model, kwargs.get("response_format"))
        _apply_groq_rate_limit(model)
        response = await acompletion(model=model, messages=messages, **kwargs)
        _record_groq_request(model)
        return self._extract_content(response)

    async def acomplete_with_fallback(
        self,
        messages: list[dict[str, str]],
        model: str | None = None,
        **kwargs: Any,
    ) -> str:
        """Async completion with one fallback model on failure. Groq models are rate-limited."""
        from litellm import acompletion
        model = model or self._default_model
        try:
            _ensure_reasoning_supported(model, kwargs.get("reasoning_effort"))
            _ensure_structured_output_supported(model, kwargs.get("response_format"))
            _apply_groq_rate_limit(model)
            response = await acompletion(model=model, messages=messages, **kwargs)
            _record_groq_request(model)
            return self._extract_content(response)
        except Exception:
            _ensure_reasoning_supported(self._fallback_model, kwargs.get("reasoning_effort"))
            _ensure_structured_output_supported(self._fallback_model, kwargs.get("response_format"))
            _apply_groq_rate_limit(self._fallback_model)
            response = await acompletion(
                model=self._fallback_model, messages=messages, **kwargs
            )
            _record_groq_request(self._fallback_model)
            return self._extract_content(response)

    @staticmethod
    def _extract_content(response: Any) -> str:
        if not response or not getattr(response, "choices", None):
            return ""
        choice = response.choices[0]
        msg = getattr(choice, "message", None)
        if not msg:
            return ""
        return getattr(msg, "content", "") or ""

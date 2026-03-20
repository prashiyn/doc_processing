"""
Wrapper around LiteLLM for completion and async completion.
Uses config/llms.yaml for default/fallback models; API keys from env (LiteLLM convention).
Groq models (model.startswith("groq/")) are rate-limited via llms.groq_ratelimit.
"""
from __future__ import annotations

from typing import Any

from doc_processing.llms.config import get_default_model, get_fallback_model


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
            _apply_groq_rate_limit(model)
            response = litellm.completion(model=model, messages=messages, **kwargs)
            _record_groq_request(model)
            return self._extract_content(response)
        except Exception:
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
            _apply_groq_rate_limit(model)
            response = await acompletion(model=model, messages=messages, **kwargs)
            _record_groq_request(model)
            return self._extract_content(response)
        except Exception:
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

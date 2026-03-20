"""
Agent runner: executes a sequence of LLM steps (e.g. extract -> classify -> summarize)
using configurable models per step. Uses LLMClient under the hood.
"""
from __future__ import annotations

from typing import Any

from doc_processing.llms.client import LLMClient
from doc_processing.llms.config import get_analysis_models


class AgentRunner:
    """
    Runs a sequence of prompt steps with optional per-step model override.
    Each step receives the previous step's output as context (or initial context).
    """

    def __init__(self, client: LLMClient | None = None):
        self._client = client or LLMClient()
        self._analysis_models = get_analysis_models()

    def run_steps(
        self,
        steps: list[dict[str, Any]],
        initial_context: str = "",
        use_analysis_models: bool = True,
    ) -> list[str]:
        """
        Run a list of steps. Each step is a dict with:
          - "prompt": str (can include {{context}} placeholder)
          - "model": optional str (override)
          - "role": optional "summarization" | "classification" | "extraction" to pick from config
        Returns list of each step's output (in order).
        """
        outputs: list[str] = []
        context = initial_context
        for step in steps:
            prompt = step.get("prompt", "").replace("{{context}}", context)
            model = step.get("model")
            if model is None and use_analysis_models:
                role = step.get("role")
                if role in self._analysis_models:
                    model = self._analysis_models[role]
            messages = [{"role": "user", "content": prompt}]
            out = self._client.complete_with_fallback(messages=messages, model=model)
            outputs.append(out)
            context = out
        return outputs

    async def arun_steps(
        self,
        steps: list[dict[str, Any]],
        initial_context: str = "",
        use_analysis_models: bool = True,
    ) -> list[str]:
        """Async version of run_steps."""
        outputs: list[str] = []
        context = initial_context
        for step in steps:
            prompt = step.get("prompt", "").replace("{{context}}", context)
            model = step.get("model")
            if model is None and use_analysis_models:
                role = step.get("role")
                if role in self._analysis_models:
                    model = self._analysis_models[role]
            messages = [{"role": "user", "content": prompt}]
            out = await self._client.acomplete_with_fallback(
                messages=messages, model=model
            )
            outputs.append(out)
            context = out
        return outputs

"""
LLM configuration: models and defaults from config_dir/llms.yaml.
API keys are not stored here; they are read from environment by LiteLLM.
"""
from __future__ import annotations

from typing import Any

import yaml

from doc_processing.config import get_config_dir


def get_llm_config() -> dict[str, Any]:
    """Load LLM config from config_dir/llms.yaml. Returns empty dict if file missing."""
    path = get_config_dir() / "llms.yaml"
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def get_default_model() -> str:
    """Default model for completions."""
    cfg = get_llm_config()
    return cfg.get("default_model", "gpt-4o-mini")


def get_fallback_model() -> str:
    """Fallback model if default fails."""
    cfg = get_llm_config()
    return cfg.get("fallback_model", "gpt-3.5-turbo")


def get_models() -> list[str]:
    """List of configured model strings (LiteLLM format)."""
    cfg = get_llm_config()
    return cfg.get("models", ["openai/gpt-4o-mini"])


def get_analysis_models() -> dict[str, str]:
    """Model per analysis task (summarization, classification, extraction)."""
    cfg = get_llm_config()
    analysis = cfg.get("analysis") or {}
    return {
        "summarization": analysis.get("summarization_model", get_default_model()),
        "classification": analysis.get("classification_model", get_default_model()),
        "extraction": analysis.get("extraction_model", get_default_model()),
    }

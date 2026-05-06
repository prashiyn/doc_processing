from __future__ import annotations

"""Config loading for doc-processing remote llm runtime."""

from functools import lru_cache
from typing import Any

import yaml

from doc_processing.config import get_config_dir, get_settings


@lru_cache(maxsize=1)
def get_llm_runtime_config() -> dict[str, Any]:
    path = get_config_dir() / "llm_config.yaml"
    if not path.exists():
        settings = get_settings()
        return {
            "service": {"base_url": settings.doc_llm_base_url, "timeout_seconds": 120},
            "use_cases": {},
        }
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def get_use_case_llm_config(use_case: str | None) -> dict[str, Any]:
    if not use_case:
        return {}
    cfg = get_llm_runtime_config()
    use_cases = cfg.get("use_cases") or {}
    raw = use_cases.get(use_case)
    return raw if isinstance(raw, dict) else {}


def get_service_llm_runtime_config() -> dict[str, Any]:
    cfg = get_llm_runtime_config()
    service = cfg.get("service")
    if not isinstance(service, dict):
        service = {}
    return service

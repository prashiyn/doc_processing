"""
Application and config directory configuration.

Single source of truth for app settings (env/.env) and config file paths.
All YAML configs (chunking, OCR, llm_runtime use-cases, etc.) live under
config_dir. Set DOC_PROCESSING_CONFIG_DIR to override; otherwise defaults to
src/config or config under cwd.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings. LLM keys and config dir from env or .env."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    app_name: str = "doc-processing"
    debug: bool = False

    # Config directory for YAML files (chunking.yaml, llm_config.yaml, glm/deepseek OCR config, etc.)
    config_dir: str | None = None

    # Temp directory for downloaded documents; files must be deleted after processing
    temp_dir: str | None = Field(default=None, validation_alias="DOC_PROCESSING_TEMP_DIR")

    # XBRL taxonomy directory for Docling XBRL backend (defaults to ./data/nse when unset)
    xbrl_taxonomy_dir: str | None = Field(default=None, validation_alias="XBRL_TAXONOMY_DIR")

    # LLM provider keys (set in .env for litellm)
    groq_api_key: str | None = None
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    tencent_secret_id: str | None = None
    tencent_secret_key: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    # Remote llm-service base URL used by doc-processing runtime client.
    doc_llm_base_url: str = Field(default="http://localhost:8001", validation_alias="DOC_LLM_BASE_URL")
    # Shared internal auth token for doc-processing -> llm-service calls.
    service_auth_token: str | None = Field(default=None, validation_alias="SERVICE_AUTH_TOKEN")


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_config_dir() -> Path:
    """
    Resolved config directory for YAML configs. Prefer DOC_PROCESSING_CONFIG_DIR;
    else first existing of cwd/src/config, cwd/config; else cwd/config.
    """
    s = get_settings()
    if s.config_dir:
        return Path(s.config_dir).resolve()
    cwd = Path.cwd()
    for candidate in (cwd / "src" / "config", cwd / "config"):
        if candidate.is_dir():
            return candidate.resolve()
    return (cwd / "config").resolve()

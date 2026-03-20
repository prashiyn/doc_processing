"""LLM completion endpoints (LiteLLM)."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from doc_processing.llms import LLMClient, get_llm_config

router = APIRouter()
_llm_client: LLMClient | None = None


def _client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


class CompletionRequest(BaseModel):
    """Request for LLM completion (LiteLLM)."""
    provider: str = Field(..., description="Provider alias: groq, ollama, openai, anthropic, tencent")
    messages: list[dict[str, str]] = Field(
        ...,
        description="Chat messages, e.g. [{\"role\": \"user\", \"content\": \"...\"}]",
        min_length=1,
    )
    model: str | None = Field(None, description="Override default model (LiteLLM model string)")


class CompletionResponse(BaseModel):
    """LLM completion content."""
    content: str = Field(..., description="Assistant reply text")


class ModelsResponse(BaseModel):
    """Configured LLM models and defaults."""
    default_model: str = Field(..., description="Default model for completions")
    fallback_model: str = Field(..., description="Fallback model on failure")
    models: list[str] = Field(..., description="List of configured model strings")


@router.post("/complete", response_model=CompletionResponse)
async def completion(req: CompletionRequest) -> CompletionResponse:
    """Get a completion from the configured LLM (with fallback)."""
    try:
        content = await _client().acomplete_with_fallback(
            req.messages, model=req.model
        )
        return CompletionResponse(content=content)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e


@router.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    """List default, fallback, and all configured LLM models."""
    cfg = get_llm_config()
    return ModelsResponse(
        default_model=cfg.get("default_model", "gpt-4o-mini"),
        fallback_model=cfg.get("fallback_model", "gpt-3.5-turbo"),
        models=cfg.get("models", []),
    )

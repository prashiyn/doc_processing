"""LLM completion endpoints (LiteLLM)."""

from __future__ import annotations

import json
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from doc_processing.llms import EmbeddingClient, LLMClient, get_llm_config

router = APIRouter()
_llm_client: LLMClient | None = None
_embedding_client: EmbeddingClient | None = None


def _client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client


def _embeddings_client() -> EmbeddingClient:
    global _embedding_client
    if _embedding_client is None:
        _embedding_client = EmbeddingClient()
    return _embedding_client


class CompletionRequest(BaseModel):
    """Request for LLM completion (LiteLLM)."""
    provider: str = Field(..., description="Provider alias: groq, ollama, openai, anthropic, tencent")
    messages: list[dict[str, str]] = Field(
        ...,
        description="Chat messages, e.g. [{\"role\": \"user\", \"content\": \"...\"}]",
        min_length=1,
    )
    model: str | None = Field(None, description="Override default model (LiteLLM model string)")
    reasoning_effort: Literal["low", "medium", "high"] | None = Field(
        None,
        description=(
            "Optional reasoning level for models that support reasoning. "
            "Allowed values: low, medium, high."
        ),
    )
    response_format: ResponseFormatJsonObject | ResponseFormatJsonSchema | None = Field(
        None,
        description=(
            "Optional structured output request (LiteLLM/OpenAI response_format). "
            "Example: {\"type\": \"json_object\"} or "
            "{\"type\": \"json_schema\", \"json_schema\": {\"name\": \"...\", \"schema\": {...}, \"strict\": true}}."
        ),
    )


class CompletionResponse(BaseModel):
    """LLM completion content."""
    content: str = Field(..., description="Assistant reply text")
    parsed: Any | None = Field(
        None,
        description="If response_format was used and content is valid JSON, this contains parsed JSON.",
    )


class JsonSchemaPayload(BaseModel):
    """Payload for response_format.type=json_schema."""

    name: str = Field(..., description="Schema name")
    schema_: dict[str, Any] = Field(
        ...,
        alias="schema",
        description="JSON Schema object",
    )
    strict: bool | None = Field(
        None,
        description="Whether to enforce strict schema adherence when supported.",
    )


class ResponseFormatJsonObject(BaseModel):
    """LiteLLM response_format for JSON object output."""

    type: Literal["json_object"] = Field(
        ...,
        description="Request valid JSON object output.",
    )


class ResponseFormatJsonSchema(BaseModel):
    """LiteLLM response_format for JSON schema output."""

    type: Literal["json_schema"] = Field(
        ...,
        description="Request output matching the provided JSON schema.",
    )
    json_schema: JsonSchemaPayload = Field(
        ...,
        description="Schema descriptor and JSON schema body.",
    )


class ModelsResponse(BaseModel):
    """Configured LLM models and defaults."""
    default_model: str = Field(..., description="Default model for completions")
    fallback_model: str = Field(..., description="Fallback model on failure")
    models: list[str] = Field(..., description="List of configured model strings")


class EmbeddingRequest(BaseModel):
    """Request for embedding generation (text/image data URIs supported by model/provider)."""

    provider: str = Field(..., description="Provider alias: openai, ollama, vertex_ai, etc.")
    input: str | list[str] = Field(
        ...,
        description=(
            "Input text or list of inputs. For image embeddings on supported models, "
            "pass base64 data URI strings."
        ),
    )
    model: str | None = Field(
        None,
        description="Override embedding model (LiteLLM model string, e.g. openai/text-embedding-3-small).",
    )
    encoding_format: Literal["float", "base64"] | None = Field(
        None,
        description="Optional embedding encoding format.",
    )
    dimensions: int | None = Field(
        None,
        ge=1,
        description="Optional output dimensions for models that support it.",
    )
    input_type: str | None = Field(
        None,
        description="Optional provider-specific input type (e.g. search_document, query, image).",
    )
    user: str | None = Field(
        None,
        description="Optional end-user identifier.",
    )


class EmbeddingDataItem(BaseModel):
    """One embedding vector item."""

    object: str = Field(..., description="Usually 'embedding'.")
    index: int = Field(..., description="Index of this embedding in the request.")
    embedding: list[float] | str = Field(
        ...,
        description="Embedding vector (float list) or base64 string depending on encoding.",
    )


class EmbeddingResponse(BaseModel):
    """Embedding response in OpenAI-compatible format."""

    object: str = Field(..., description="Usually 'list'.")
    model: str = Field(..., description="Model used to generate embeddings.")
    data: list[EmbeddingDataItem] = Field(..., description="Embedding vectors.")
    usage: dict[str, Any] | None = Field(
        None,
        description="Optional token usage stats if provided by backend.",
    )


@router.post("/complete", response_model=CompletionResponse)
async def completion(req: CompletionRequest) -> CompletionResponse:
    """Get a completion from the configured LLM (with fallback). Used only for document processing."""
    try:
        content = await _client().acomplete_with_fallback(
            req.messages,
            model=req.model,
            reasoning_effort=req.reasoning_effort,
            response_format=req.response_format,
        )
        parsed: Any | None = None
        if req.response_format is not None:
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = None
        return CompletionResponse(content=content, parsed=parsed)
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


@router.post("/embeddings", response_model=EmbeddingResponse)
async def embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    """Generate embeddings for text/image inputs via LiteLLM or direct Ollama fallback."""
    try:
        result = await _embeddings_client().aembed(
            req.input,
            model=req.model,
            encoding_format=req.encoding_format,
            dimensions=req.dimensions,
            input_type=req.input_type,
            user=req.user,
        )
        return EmbeddingResponse(
            object=str(result.get("object", "list")),
            model=str(result.get("model", req.model or "")),
            data=[
                EmbeddingDataItem(
                    object=str(item.get("object", "embedding")),
                    index=int(item.get("index", idx)),
                    embedding=item.get("embedding"),
                )
                for idx, item in enumerate(result.get("data", []))
            ],
            usage=result.get("usage"),
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e)) from e

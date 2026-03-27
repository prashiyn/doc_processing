"""Tests for /llm endpoints (LLM client mocked where needed)."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient


def test_llm_models_lists_config(client: TestClient) -> None:
    r = client.get("/llm/models")
    assert r.status_code == 200
    body = r.json()
    assert "default_model" in body
    assert "fallback_model" in body
    assert isinstance(body["models"], list)
    assert len(body["models"]) >= 1


def test_llm_complete_success_mocked(client: TestClient) -> None:
    mock_llm = MagicMock()
    mock_llm.acomplete_with_fallback = AsyncMock(return_value="mocked assistant reply")

    with patch("doc_processing.routers.llm._client", return_value=mock_llm):
        r = client.post(
            "/llm/complete",
            json={
                "provider": "openai",
                "messages": [{"role": "user", "content": "ping"}],
            },
        )

    assert r.status_code == 200
    assert r.json() == {"content": "mocked assistant reply", "parsed": None}
    mock_llm.acomplete_with_fallback.assert_awaited_once()


def test_llm_complete_structured_json_object_mocked(client: TestClient) -> None:
    mock_llm = MagicMock()
    mock_llm.acomplete_with_fallback = AsyncMock(return_value='{"ok": true, "n": 1}')

    with patch("doc_processing.routers.llm._client", return_value=mock_llm):
        r = client.post(
            "/llm/complete",
            json={
                "provider": "openai",
                "messages": [{"role": "user", "content": "ping"}],
                "response_format": {"type": "json_object"},
            },
        )

    assert r.status_code == 200
    body = r.json()
    assert body["content"] == '{"ok": true, "n": 1}'
    assert body["parsed"] == {"ok": True, "n": 1}


def test_llm_complete_reasoning_effort_passthrough(client: TestClient) -> None:
    mock_llm = MagicMock()
    mock_llm.acomplete_with_fallback = AsyncMock(return_value="reasoned reply")

    with patch("doc_processing.routers.llm._client", return_value=mock_llm):
        r = client.post(
            "/llm/complete",
            json={
                "provider": "openai",
                "messages": [{"role": "user", "content": "ping"}],
                "reasoning_effort": "low",
            },
        )

    assert r.status_code == 200
    assert r.json() == {"content": "reasoned reply", "parsed": None}
    _, kwargs = mock_llm.acomplete_with_fallback.await_args
    assert kwargs["reasoning_effort"] == "low"


def test_llm_complete_reasoning_effort_invalid_value(client: TestClient) -> None:
    r = client.post(
        "/llm/complete",
        json={
            "provider": "openai",
            "messages": [{"role": "user", "content": "ping"}],
            "reasoning_effort": "extreme",
        },
    )
    assert r.status_code == 422


def test_llm_complete_validation_empty_messages(client: TestClient) -> None:
    r = client.post(
        "/llm/complete",
        json={"provider": "openai", "messages": []},
    )
    assert r.status_code == 422


def test_llm_embeddings_success_mocked(client: TestClient) -> None:
    mock_embed = MagicMock()
    mock_embed.aembed = AsyncMock(
        return_value={
            "object": "list",
            "model": "openai/text-embedding-3-small",
            "data": [
                {"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]},
            ],
            "usage": {"prompt_tokens": 5, "total_tokens": 5},
        }
    )

    with patch("doc_processing.routers.llm._embeddings_client", return_value=mock_embed):
        r = client.post(
            "/llm/embeddings",
            json={
                "provider": "openai",
                "input": ["hello world"],
                "model": "openai/text-embedding-3-small",
            },
        )

    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    assert body["model"] == "openai/text-embedding-3-small"
    assert len(body["data"]) == 1
    assert body["data"][0]["index"] == 0
    assert body["usage"]["total_tokens"] == 5


def test_llm_embeddings_image_data_uri_mocked(client: TestClient) -> None:
    mock_embed = MagicMock()
    mock_embed.aembed = AsyncMock(
        return_value={
            "object": "list",
            "model": "cohere/embed-english-v3.0",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.4, 0.5]}],
            "usage": None,
        }
    )
    image_data_uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"

    with patch("doc_processing.routers.llm._embeddings_client", return_value=mock_embed):
        r = client.post(
            "/llm/embeddings",
            json={
                "provider": "cohere",
                "input": [image_data_uri],
                "model": "cohere/embed-english-v3.0",
            },
        )

    assert r.status_code == 200
    assert r.json()["data"][0]["embedding"] == [0.4, 0.5]

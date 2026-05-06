"""Service boundary tests: /llm routes are not exposed by doc-processing."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_llm_models_not_exposed_in_doc_processing(client: TestClient) -> None:
    r = client.get("/llm/models")
    assert r.status_code == 404


def test_llm_complete_not_exposed_in_doc_processing(client: TestClient) -> None:
    r = client.post(
        "/llm/complete",
        json={
            "provider": "openai",
            "messages": [{"role": "user", "content": "ping"}],
        },
    )
    assert r.status_code == 404


def test_llm_embeddings_not_exposed_in_doc_processing(client: TestClient) -> None:
    r = client.post(
        "/llm/embeddings",
        json={
            "provider": "openai",
            "input": ["hello"],
            "model": "openai/text-embedding-3-small",
        },
    )
    assert r.status_code == 404

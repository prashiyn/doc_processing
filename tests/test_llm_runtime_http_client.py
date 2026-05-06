from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from doc_processing.llm_runtime.http_client import HttpLLMRuntime


def _mock_response(body: dict, status_code: int = 200) -> MagicMock:
    r = MagicMock()
    r.status_code = status_code
    r.json.return_value = body
    if status_code >= 400:
        import requests

        r.raise_for_status.side_effect = requests.HTTPError(f"status {status_code}")
    else:
        r.raise_for_status.return_value = None
    return r


@patch("doc_processing.llm_runtime.http_client.get_use_case_llm_config")
@patch("doc_processing.llm_runtime.http_client.requests.post")
def test_complete_uses_use_case_model_and_provider(mock_post: MagicMock, mock_case_cfg: MagicMock) -> None:
    mock_case_cfg.return_value = {"provider": "openai", "model": "openai/gpt-4o-mini"}
    mock_post.return_value = _mock_response({"content": "ok"})

    c = HttpLLMRuntime(base_url="http://llm.local", service_auth_token="tok")
    out = c.complete_with_fallback(messages=[{"role": "user", "content": "ping"}], use_case="chunk_coreference")
    assert out == "ok"

    _, kwargs = mock_post.call_args
    assert kwargs["json"]["provider"] == "openai"
    assert kwargs["json"]["model"] == "openai/gpt-4o-mini"
    assert kwargs["headers"]["X-Service-Token"] == "tok"


@patch("doc_processing.llm_runtime.http_client.get_use_case_llm_config")
@patch("doc_processing.llm_runtime.http_client.requests.post")
def test_complete_explicit_model_overrides_use_case(mock_post: MagicMock, mock_case_cfg: MagicMock) -> None:
    mock_case_cfg.return_value = {"provider": "openai", "model": "openai/gpt-4o-mini"}
    mock_post.return_value = _mock_response({"content": "ok"})
    c = HttpLLMRuntime(base_url="http://llm.local")

    _ = c.complete_with_fallback(
        messages=[{"role": "user", "content": "ping"}],
        use_case="chunk_coreference",
        model="groq/llama-3.3-70b-versatile",
    )
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == "groq/llama-3.3-70b-versatile"
    assert kwargs["json"]["provider"] == "groq"


@patch("doc_processing.llm_runtime.http_client.get_use_case_llm_config")
@patch("doc_processing.llm_runtime.http_client.requests.post")
def test_embed_uses_use_case_defaults(mock_post: MagicMock, mock_case_cfg: MagicMock) -> None:
    mock_case_cfg.return_value = {"provider": "openai", "model": "openai/text-embedding-3-small"}
    mock_post.return_value = _mock_response({"object": "list", "model": "openai/text-embedding-3-small", "data": []})

    c = HttpLLMRuntime(base_url="http://llm.local")
    out = c.embed(input_data=["hello"], use_case="embedding_document")
    assert out["object"] == "list"
    _, kwargs = mock_post.call_args
    assert kwargs["json"]["model"] == "openai/text-embedding-3-small"
    assert kwargs["json"]["provider"] == "openai"


@patch("doc_processing.llm_runtime.http_client.requests.post")
def test_complete_raises_runtime_error_on_http_failure(mock_post: MagicMock) -> None:
    import requests

    mock_post.side_effect = requests.RequestException("boom")
    c = HttpLLMRuntime(base_url="http://llm.local")
    with pytest.raises(RuntimeError, match="completion call failed"):
        c.complete_with_fallback(messages=[{"role": "user", "content": "x"}])

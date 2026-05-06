"""Tests for /documents endpoints (validation + mocked heavy pipelines)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient


def _sample_ffp_chunks() -> list[dict]:
    return [
        {
            "chunk_id": "c1",
            "content": "Hello world",
            "type": "text",
            "doc_id": "readme",
            "page": 1,
            "bundle_id": "b0",
            "section_title": "Intro",
            "title_summary": "",
            "publish_date": None,
            "prev_chunk": None,
            "next_chunk": None,
            "references": [],
        }
    ]


@pytest.mark.parametrize(
    "path",
    [
        "/documents/convert-ms-docs",
        "/documents/process-using-unstructured",
        "/documents/pdf-to-markdown",
        "/documents/xbrl-to-markdown",
        "/documents/process-using-deepseek",
        "/documents/process-using-glm",
        "/documents/chunk",
    ],
)
def test_documents_missing_file_and_url_returns_422(client: TestClient, path: str) -> None:
    r = client.post(path)
    assert r.status_code == 422
    detail = r.json().get("detail", "")
    text = str(detail).lower()
    assert "file" in text or "url" in text or "field required" in text


def test_chunk_empty_upload_returns_422(client: TestClient) -> None:
    r = client.post(
        "/documents/chunk",
        files={"file": ("empty.md", b"", "text/markdown")},
    )
    assert r.status_code == 422


def test_chunk_unsupported_extension_returns_415(client: TestClient) -> None:
    r = client.post(
        "/documents/chunk",
        files={"file": ("notes.txt", b"plain", "text/plain")},
    )
    assert r.status_code == 415


def test_chunk_markdown_upload_mocked_pipeline(client: TestClient) -> None:
    samples = _sample_ffp_chunks()
    with patch(
        "doc_processing.routers.documents._ffp_chunk_document_sync",
        return_value=samples,
    ) as mock_run:
        r = client.post(
            "/documents/chunk",
            files={"file": ("readme.md", b"# Title\n\nBody.", "text/markdown")},
        )

    assert r.status_code == 200
    body = r.json()
    assert "chunks" in body
    assert len(body["chunks"]) == 1
    assert body["chunks"][0]["chunk_id"] == "c1"
    assert body["chunks"][0]["type"] == "text"
    mock_run.assert_called_once()
    call_kw = mock_run.call_args.kwargs
    assert call_kw["ext"] == ".md"
    assert call_kw["doc_id"] == "readme"


def test_chunk_url_query_mocked_fetch_and_pipeline(
    client: TestClient, tmp_path: Path
) -> None:
    samples = _sample_ffp_chunks()
    fake_file = tmp_path / "remote.md"
    fake_file.write_bytes(b"# From URL\n")

    with (
        patch(
            "doc_processing.routers.documents._source_from_url_to_temp",
            new_callable=AsyncMock,
            return_value=fake_file,
        ),
        patch(
            "doc_processing.routers.documents._ffp_chunk_document_sync",
            return_value=samples,
        ) as mock_run,
    ):
        r = client.post(
            "/documents/chunk",
            params={"url": "https://example.com/docs/report.md"},
        )

    assert r.status_code == 200
    assert len(r.json()["chunks"]) == 1
    mock_run.assert_called_once()
    args, kwargs = mock_run.call_args
    assert args[0] == fake_file
    assert kwargs["ext"] == ".md"


def test_convert_ms_docs_unsupported_type_returns_415(client: TestClient) -> None:
    r = client.post(
        "/documents/convert-ms-docs",
        files={"file": ("readme.txt", b"hello", "text/plain")},
    )
    assert r.status_code == 415


def test_pdf_to_markdown_validation_only_no_crash(client: TestClient) -> None:
    """Without a real PDF pipeline, only assert missing-input is rejected."""
    r = client.post("/documents/pdf-to-markdown")
    assert r.status_code == 422

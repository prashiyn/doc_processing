"""Shared pytest fixtures for API tests."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from doc_processing.main import create_app


@pytest.fixture
def client() -> TestClient:
    """Synchronous TestClient over a fresh FastAPI app instance."""
    return TestClient(create_app())

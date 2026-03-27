"""Tests for /health endpoints."""

from __future__ import annotations

from fastapi.testclient import TestClient


def test_health_liveness(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_health_readiness(client: TestClient) -> None:
    r = client.get("/health/ready")
    assert r.status_code == 200
    assert r.json() == {"status": "ready"}

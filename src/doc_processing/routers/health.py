"""Health check endpoints."""

from fastapi import APIRouter

router = APIRouter()


@router.get("", summary="Liveness")
async def health() -> dict[str, str]:
    """Simple liveness check. Returns 200 when the service is up."""
    return {"status": "ok"}


@router.get("/ready", summary="Readiness")
async def ready() -> dict[str, str]:
    """Readiness check. Use for k8s/load-balancer probes."""
    return {"status": "ready"}

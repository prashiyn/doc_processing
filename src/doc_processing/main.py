"""FastAPI application entrypoint."""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from doc_processing.config import get_settings
from doc_processing.routers import documents, health, llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hooks."""
    yield
    # Teardown if needed


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        description="Document processing, conversion to markdown, chunking for RAG, and LLM completions.",
        version="0.1.0",
        debug=settings.debug,
        lifespan=lifespan,
    )
    app.include_router(health.router, prefix="/health", tags=["health"])
    app.include_router(documents.router, prefix="/documents", tags=["documents"])
    app.include_router(llm.router, prefix="/llm", tags=["llm"])
    return app


app = create_app()


def run() -> None:
    import uvicorn
    uvicorn.run("doc_processing.main:app", host="0.0.0.0", port=8000, reload=True)

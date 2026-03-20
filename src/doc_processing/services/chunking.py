"""Text chunking for RAG: fixed-size with overlap."""

from __future__ import annotations


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[str]:
    """Split text into overlapping chunks for embedding/retrieval."""
    if not text or chunk_size <= 0:
        return []
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size - 1)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = end - chunk_overlap
    return [c for c in chunks if c]

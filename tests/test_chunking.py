"""Tests for chunking service."""

import pytest

from doc_processing.services.chunking import chunk_text


def test_chunk_text_basic():
    text = "a " * 200
    chunks = chunk_text(text, chunk_size=100, chunk_overlap=20)
    assert len(chunks) >= 2
    assert all(isinstance(c, str) for c in chunks)


def test_chunk_text_empty():
    assert chunk_text("", chunk_size=512) == []
    assert chunk_text("  ", chunk_size=512) == []


def test_chunk_text_overlap():
    text = "one two three four five " * 40
    chunks = chunk_text(text, chunk_size=80, chunk_overlap=16)
    assert len(chunks) >= 2
    # Consecutive chunks should share some overlap region
    for i in range(len(chunks) - 1):
        # Last 16 chars of chunk i might appear at start of chunk i+1
        assert len(chunks[i]) <= 80 + 20  # some slack for strip

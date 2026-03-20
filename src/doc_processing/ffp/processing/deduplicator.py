from __future__ import annotations

"""
Chunk-level deduplication using TF-IDF + cosine similarity.

The similarity threshold is taken from `config/chunking.yaml` (params.dedup_similarity_threshold).
"""

from typing import Any, Iterable

import yaml
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from doc_processing.config import get_config_dir


def _load_dedup_threshold() -> float:
    path = get_config_dir() / "chunking.yaml"
    if not path.exists():
        return 0.7
    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    params = cfg.get("params") or {}
    try:
        return float(params.get("dedup_similarity_threshold", 0.7))
    except (TypeError, ValueError):
        return 0.7


class Deduplicator:
    """Remove near-duplicate chunks based on cosine similarity."""

    def __init__(self, threshold: float | None = None) -> None:
        self.threshold = threshold if threshold is not None else _load_dedup_threshold()

    def run(self, chunks: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a pruned list of chunks with duplicates removed."""
        chunks_list = list(chunks)
        if not chunks_list:
            return []

        texts = [c.get("content") or "" for c in chunks_list]
        if len(texts) == 1:
            return chunks_list

        vec = TfidfVectorizer()
        matrix = vec.fit_transform(texts)

        keep_indices: list[int] = []
        for i in range(len(chunks_list)):
            duplicate = False
            for j in keep_indices:
                sim = cosine_similarity(matrix[i], matrix[j])[0][0]
                if sim > self.threshold:
                    duplicate = True
                    break
            if not duplicate:
                keep_indices.append(i)

        return [chunks_list[i] for i in keep_indices]


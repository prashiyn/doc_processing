from __future__ import annotations

"""
Bundle-aware chunk segmentation.

Starting from MinerU/markdown blocks, we:
- group blocks into bundles based on section titles (`level == 1` by convention)
- within each bundle, create chunks with content and type (text/table/image)
"""

from dataclasses import dataclass
from typing import Any, Iterable

import uuid


Block = dict[str, Any]

# Canonical chunk content types (must match pipeline Chunk.type).
CHUNK_TYPE_TEXT = "text"
CHUNK_TYPE_TABLE = "table"
CHUNK_TYPE_IMAGE = "image"


def _normalize_block_type(t: Any) -> str:
    s = (t or "text").lower().strip()
    if s in (CHUNK_TYPE_TABLE, CHUNK_TYPE_IMAGE):
        return s
    return CHUNK_TYPE_TEXT


@dataclass
class _Segment:
    """One piece of content in a bundle: (content, type, page)."""
    content: str
    type: str
    page: int | None


@dataclass
class Bundle:
    bundle_id: str
    section_title: str | None
    segments: list[_Segment]


class ChunkSegmenter:
    """Segment blocks into bundles and size-bounded chunks (content + type)."""

    def __init__(self, max_chars: int = 900, min_block_chars: int = 200) -> None:
        self.max_chars = max_chars
        self.min_block_chars = min_block_chars

    def build_bundles(self, blocks: Iterable[Block]) -> list[Bundle]:
        """Group blocks into bundles; each segment keeps content, type, and page."""
        bundles: list[Bundle] = []
        current_segments: list[_Segment] = []
        current_bundle_id: str | None = None
        current_title: str | None = None

        for block in blocks:
            content = (block.get("text") or "").strip()
            level = block.get("level")
            if not content:
                continue

            block_type = _normalize_block_type(block.get("type"))
            page = block.get("page")
            try:
                page_int = int(page) if page is not None else None
            except (TypeError, ValueError):
                page_int = None

            # new top-level section -> close previous bundle
            if level == 1:
                if current_segments:
                    bundles.append(
                        Bundle(
                            bundle_id=current_bundle_id or str(uuid.uuid4()),
                            section_title=current_title,
                            segments=list(current_segments),
                        )
                    )
                current_bundle_id = str(uuid.uuid4())
                current_title = content
                current_segments = []
                continue

            current_segments.append(_Segment(content=content, type=block_type, page=page_int))

        if current_segments:
            bundles.append(
                Bundle(
                    bundle_id=current_bundle_id or str(uuid.uuid4()),
                    section_title=current_title,
                    segments=list(current_segments),
                )
            )

        return bundles

    def split_bundle_into_chunks(self, bundle: Bundle) -> list[dict[str, Any]]:
        """Split a bundle into chunks; each chunk has content, type (from first segment), page."""
        chunks: list[dict[str, Any]] = []
        buffer = ""
        buffer_type: str = CHUNK_TYPE_TEXT
        buffer_page: int | None = None

        for seg in bundle.segments:
            candidate = (buffer + " " + seg.content).strip() if buffer else seg.content
            if len(candidate) <= self.max_chars:
                if not buffer:
                    buffer_type = seg.type
                    buffer_page = seg.page
                buffer = candidate
                continue

            if buffer:
                chunks.append(
                    {
                        "bundle_id": bundle.bundle_id,
                        "section_title": bundle.section_title,
                        "content": buffer.strip(),
                        "type": buffer_type,
                        "page": buffer_page,
                    }
                )
            buffer = seg.content
            buffer_type = seg.type
            buffer_page = seg.page

        if buffer:
            chunks.append(
                {
                    "bundle_id": bundle.bundle_id,
                    "section_title": bundle.section_title,
                    "content": buffer.strip(),
                    "type": buffer_type,
                    "page": buffer_page,
                }
            )

        return chunks

    def segment(self, blocks: Iterable[Block]) -> list[dict[str, Any]]:
        """Blocks → chunk dicts with bundle_id, section_title, content, type, page."""
        bundles = self.build_bundles(blocks)
        out: list[dict[str, Any]] = []
        for bundle in bundles:
            out.extend(self.split_bundle_into_chunks(bundle))
        return out


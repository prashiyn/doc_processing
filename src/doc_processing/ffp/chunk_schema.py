"""
OpenAPI / Pydantic models for FFP chunks.

`ChunkItem` is derived from the `Chunk` dataclass field-for-field, with an explicit
schema for `references` (see docs/REFERENCES.md) so OpenAPI is not a bare
`list[object]`.
"""

from __future__ import annotations

import dataclasses
from functools import lru_cache
from typing import Any, Literal, get_type_hints

from pydantic import BaseModel, Field, create_model

from doc_processing.ffp.pipeline import Chunk

ReferenceTypeLiteral = Literal[
    "TABLE",
    "FIGURE",
    "SECTION",
    "SUBSECTION",
    "APPENDIX",
    "NOTE",
    "FOOTNOTE",
    "SCHEDULE",
    "STATEMENT",
    "REGULATION",
    "OTHER",
]


class ChunkReferenceItem(BaseModel):
    """One extracted cross-reference (aligned with docs/REFERENCES.md)."""

    reference_text: str = Field(..., description="Exact phrase as it appears in the chunk")
    reference_type: ReferenceTypeLiteral | str = Field(
        ...,
        description="TABLE, FIGURE, SECTION, SUBSECTION, APPENDIX, NOTE, FOOTNOTE, "
        "SCHEDULE, STATEMENT, REGULATION, OTHER",
    )
    target_label: str = Field(..., description="Normalized identifier for the target")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence 0–1")


@lru_cache(maxsize=1)
def get_chunk_item_model() -> type[BaseModel]:
    """Build a Pydantic model matching `Chunk` for API responses."""
    hints = get_type_hints(Chunk)
    defs: dict[str, Any] = {}

    for f in dataclasses.fields(Chunk):
        if f.name == "references":
            defs[f.name] = (
                list[ChunkReferenceItem],
                Field(default_factory=list, description="Cross-references extracted from chunk text"),
            )
            continue

        ann = hints.get(f.name, Any)
        if f.default_factory is not dataclasses.MISSING:
            defs[f.name] = (ann, Field(default_factory=f.default_factory))
        elif f.default is not dataclasses.MISSING:
            defs[f.name] = (ann, Field(default=f.default))
        else:
            defs[f.name] = (ann, ...)

    return create_model(
        "ChunkItem",
        __base__=BaseModel,
        __doc__="One FFP chunk; fields mirror `doc_processing.ffp.pipeline.Chunk`.",
        **defs,
    )

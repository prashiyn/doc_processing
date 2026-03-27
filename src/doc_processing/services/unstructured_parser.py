"""File parser using the Unstructured pipeline.

Extracts tables, images, and full metadata for PDF, image, and Markdown files.
Uses hi_res strategy for table extraction (see table-extraction-from-pdf and
extract-image-block-types in Unstructured docs). For PDFs: partition_pdf with
skip_infer_table_types=False and optional extract_image_block_types=["Image", "Table"].
For images: partition_image with strategy="hi_res". For Markdown: partition (auto).
"""
from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from unstructured.partition.auto import partition
from unstructured.partition.image import partition_image
from unstructured.partition.pdf import partition_pdf


@dataclass
class TableBlock:
    text: str
    text_as_html: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageBlock:
    metadata: dict[str, Any] = field(default_factory=dict)
    image_base64: str | None = None


@dataclass
class FileProcessorResult:
    tables: list[TableBlock] = field(default_factory=list)
    images: list[ImageBlock] = field(default_factory=list)
    extract_output_dir: Path | None = None


def _element_metadata_to_dict(el: Any) -> dict[str, Any]:
    meta = getattr(el, "metadata", None)
    if meta is None:
        return {}
    out: dict[str, Any] = {}
    for key in dir(meta):
        if key.startswith("_"):
            continue
        try:
            val = getattr(meta, key)
            if callable(val):
                continue
            out[key] = str(val) if not isinstance(val, (str, int, float, bool, type(None))) and hasattr(val, "__dict__") else val
        except Exception:
            continue
    return out


def _element_to_table_block(el: Any) -> TableBlock:
    meta = _element_metadata_to_dict(el)
    th = meta.get("text_as_html")
    text_as_html = th if isinstance(th, str) else (getattr(el.metadata, "text_as_html", None) if hasattr(el, "metadata") and hasattr(el.metadata, "text_as_html") else None)
    return TableBlock(text=getattr(el, "text", "") or "", text_as_html=text_as_html, metadata=meta)


def _element_to_image_block(el: Any) -> ImageBlock:
    meta = _element_metadata_to_dict(el)
    b64 = meta.get("image_base64")
    image_base64 = b64 if isinstance(b64, str) else (getattr(el.metadata, "image_base64", None) if hasattr(el, "metadata") and hasattr(el.metadata, "image_base64") else None)
    return ImageBlock(metadata=meta, image_base64=image_base64)


def process_using_unstructured(
    file_content: bytes | str | Path,
    file_type: Literal["pdf", "image", "markdown"] | None = None,
    *,
    strategy: str = "hi_res",
    extract_image_block_types: list[str] | None = None,
    extract_image_block_to_payload: bool = True,
    extract_image_block_output_dir: str | Path | None = None,
    skip_infer_table_types: bool = False,
) -> FileProcessorResult:
    """Extract tables, images, and their metadata from a PDF, image, or Markdown file.

    Uses Unstructured hi_res pipeline. For PDF: partition_pdf with skip_infer_table_types=False
    and extract_image_block_types for embedded images/tables. For image files: partition_image
    with strategy="hi_res". For Markdown: partition() (tables from markdown syntax).

    Args:
        file_content: Path (str/Path) to file or raw bytes (then file_type required).
        file_type: "pdf" | "image" | "markdown". Inferred from path suffix when path given.
        strategy: Partition strategy; "hi_res" recommended for table extraction.
        extract_image_block_types: Element types to extract as images; default ["Image", "Table"].
        extract_image_block_to_payload: If True, store extracted blocks as base64 in metadata.
        extract_image_block_output_dir: If set and to_payload False, save blocks to this dir.
        skip_infer_table_types: If False (default), infer table types for PDF table extraction.

    Returns:
        FileProcessorResult with .tables (TableBlock with text, text_as_html, metadata),
        .images (ImageBlock with metadata and optional image_base64), .extract_output_dir.
    """
    extract_image_block_types = extract_image_block_types or ["Image", "Table"]
    path: Path
    inferred_type: Literal["pdf", "image", "markdown"] | None = None
    if isinstance(file_content, Path):
        path = file_content
        s = path.suffix.lower()
        if file_type is None:
            inferred_type = "pdf" if s == ".pdf" else "markdown" if s in (".md", ".markdown") else "image" if s in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif") else None
        else:
            inferred_type = file_type
    elif isinstance(file_content, str):
        path = Path(file_content)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        s = path.suffix.lower()
        if file_type is None:
            inferred_type = "pdf" if s == ".pdf" else "markdown" if s in (".md", ".markdown") else "image" if s in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif") else None
        else:
            inferred_type = file_type
    else:
        if file_type is None:
            raise ValueError("file_type must be provided when file_content is bytes")
        inferred_type = file_type
        suffix = {"pdf": ".pdf", "image": ".png", "markdown": ".md"}.get(file_type, ".bin")
        tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        tmp.write(file_content)
        tmp.close()
        path = Path(tmp.name)
    kind = inferred_type or ("pdf" if path.suffix.lower() == ".pdf" else "markdown" if path.suffix.lower() in (".md", ".markdown") else "image")
    try:
        if kind == "pdf":
            elements = partition_pdf(
                filename=str(path),
                strategy=strategy,
                skip_infer_table_types=skip_infer_table_types,
                extract_image_block_types=extract_image_block_types,
                extract_image_block_to_payload=extract_image_block_to_payload,
                extract_image_block_output_dir=str(extract_image_block_output_dir) if extract_image_block_output_dir else None,
            )
        elif kind == "image":
            elements = partition_image(
                filename=str(path),
                strategy=strategy,
                extract_image_block_types=extract_image_block_types,
                extract_image_block_to_payload=extract_image_block_to_payload,
                extract_image_block_output_dir=str(extract_image_block_output_dir) if extract_image_block_output_dir else None,
            )
        else:
            elements = partition(filename=str(path), strategy=strategy)
        tables = [_element_to_table_block(el) for el in elements if getattr(el, "category", None) == "Table"]
        images = [_element_to_image_block(el) for el in elements if getattr(el, "category", None) == "Image"]
        for el in elements:
            if getattr(el, "category", None) == "Table" and getattr(getattr(el, "metadata", None), "image_base64", None):
                images.append(_element_to_image_block(el))
        out_dir = Path(extract_image_block_output_dir) if extract_image_block_output_dir else None
        return FileProcessorResult(tables=tables, images=images, extract_output_dir=out_dir)
    finally:
        if isinstance(file_content, bytes) and path.exists():
            path.unlink(missing_ok=True)


# Backward-compatible alias while callsites migrate.
process_file = process_using_unstructured

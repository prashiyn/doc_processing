"""
Document converters.

File-to-markdown (markitdown, docling), iXBRL, YouTube transcript,
and markdown tables to DataFrames.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Literal
from lxml import etree as ET
from markitdown import MarkItDown
from ixbrl_parse.ixbrl import parse

from mistletoe.block_token import Table, TableCell, TableRow


import pandas as pd

from mistletoe import Document


def ixbrl_to_format(
    xml_string: str,
    output_format: Literal["json", "dict", "markdown"],
) -> str | dict[str, Any]:
    """
    Parse iXBRL/XBRL XML string and convert to the requested format.
    Uses ixbrl-parse (https://github.com/cybermaggedon/ixbrl-parse).

    Args:
        xml_string: Raw iXBRL or XBRL XML content.
        output_format: One of "json", "dict", "markdown".

    Returns:
        For "json": JSON string. For "dict": nested dict. For "markdown": markdown string.
        On parse failure returns empty dict, "{}", or "" depending on format.
    """

    try:
        raw = xml_string.encode("utf-8") if isinstance(xml_string, str) else xml_string
        tree = ET.ElementTree(ET.fromstring(raw))
        ix = parse(tree)
        data = ix.to_dict()
        flat = ix.flatten()
    except Exception:
        if output_format == "dict":
            return {}
        if output_format == "json":
            return "{}"
        return ""

    if output_format == "dict":
        return data
    if output_format == "json":
        return json.dumps(data, default=str)
    if output_format == "markdown":
        lines = ["# iXBRL data", ""]
        if flat.get("contexts"):
            lines.append("## Contexts")
            for ctx in flat["contexts"]:
                lines.append(f"- {ctx}")
            lines.append("")
        if flat.get("values"):
            lines.append("## Values")
            for v in flat["values"]:
                name = v.get("name", "")
                ctx = v.get("context", "")
                val = v.get("value", "")
                lines.append(f"- **{name}** (context: {ctx}): {val}")
        return "\n".join(lines) if len(lines) > 2 else json.dumps(data, default=str, indent=2)
    raise ValueError(f"output_format must be 'json', 'dict', or 'markdown'; got {output_format!r}")


def file_to_markdown_using_markitdown(
    source: str | Path | bytes,
    file_extension: str | None = None,
) -> str:
    """
    Convert a supported file (PDF, DOCX, XLSX, HTML, XML, etc.) to markdown.
    Uses markitdown (https://github.com/microsoft/markitdown). Install markitdown[all]
    for all format support.

    Args:
        source: File path (str or Path) or raw bytes of the file content.
        file_extension: Required if source is bytes (e.g. ".pdf", ".xml", ".html").

    Returns:
        Markdown string. Empty string on failure.
    """

    md = MarkItDown()
    if isinstance(source, bytes):
        if not file_extension or not file_extension.startswith("."):
            file_extension = ".bin"
        suffix = file_extension if file_extension.startswith(".") else f".{file_extension}"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(source)
            path = f.name
        try:
            result = md.convert(path)
            return (result.text_content or "").strip()
        finally:
            Path(path).unlink(missing_ok=True)
    path = Path(source)
    if not path.exists():
        return ""
    result = md.convert(str(path))
    return (result.text_content or "").strip()


def _mistletoe_span_to_text(token: Any) -> str:
    """Recursively extract plain text from a mistletoe span-level token (e.g. inside a TableCell)."""
    children = getattr(token, "children", None) or []
    if hasattr(token, "content") and not children:
        return getattr(token, "content", "") or ""
    return "".join(_mistletoe_span_to_text(c) for c in children)


def _mistletoe_table_to_dataframe(table: Any) -> "pd.DataFrame":
    """Convert a mistletoe Table block token to a pandas DataFrame."""
    if not isinstance(table, Table):
        return pd.DataFrame()
    header_row = getattr(table, "header", None)
    body_rows = list(getattr(table, "children", []))
    if header_row is None and body_rows:
        header_row = body_rows[0]
        body_rows = body_rows[1:]
    if header_row is None:
        return pd.DataFrame()
    header_cells = getattr(header_row, "children", [])
    columns = [_mistletoe_span_to_text(c) for c in header_cells if isinstance(c, TableCell)]
    if not columns:
        return pd.DataFrame()
    data = []
    for row in body_rows:
        if not isinstance(row, TableRow):
            continue
        cells = getattr(row, "children", [])
        row_texts = [_mistletoe_span_to_text(c) for c in cells if isinstance(c, TableCell)]
        if len(row_texts) <= len(columns):
            data.append(row_texts + [""] * (len(columns) - len(row_texts)))
        else:
            data.append(row_texts[: len(columns)])
    return pd.DataFrame(data, columns=columns)


def _mistletoe_collect_tables(node: Any, out: list[Any]) -> None:
    """Recursively collect all Table block tokens from a mistletoe document tree."""

    if isinstance(node, Table):
        out.append(node)
    children = getattr(node, "children", None) or []
    for child in children:
        _mistletoe_collect_tables(child, out)


def markdown_tables_to_dataframes(
    source: str | Path,
) -> list[pd.DataFrame]:
    """
    Extract all markdown tables from a file or string and return a list of pandas DataFrames.

    Uses mistletoe (https://github.com/miyuchina/mistletoe) to parse markdown into an AST,
    then collects all Table block tokens (including tables inside block quotes) and converts
    each to a DataFrame. Non-table content in the markdown is ignored.

    Args:
        source: Path to a markdown file, or a string containing markdown.

    Returns:
        List of DataFrames, one per table, in document order. Empty list if no tables
        or on parse error.
    """
    if isinstance(source, Path):
        source = source.read_text(encoding="utf-8")
    text = source if isinstance(source, str) else ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip():
        return []
    try:
        doc = Document(text)
    except Exception:
        return []
    tables: list[Any] = []
    _mistletoe_collect_tables(doc, tables)
    return [_mistletoe_table_to_dataframe(t) for t in tables]


def youtube_url_to_transcript(youtube_url: str) -> str:
    """
    Extract the transcript of a YouTube video as markdown.

    Uses MarkItDown's YouTube transcription support
    (https://github.com/microsoft/markitdown); requires markitdown[all] or
    markitdown[youtube-transcription]. Useful for analyst calls, earnings
    calls, and other video links.

    Args:
        youtube_url: Full YouTube URL (e.g. https://www.youtube.com/watch?v=VIDEO_ID
                     or https://youtu.be/VIDEO_ID).

    Returns:
        Transcript content in markdown. Empty string if the URL is not
        recognized as YouTube, fetch fails, or the video has no transcript.
    """
    url = (youtube_url or "").strip()
    if not url:
        return ""
    if "youtube.com" not in url and "youtu.be" not in url:
        return ""
    try:
        md = MarkItDown()
        result = md.convert(url)
        return (result.text_content or "").strip()
    except Exception:
        return ""



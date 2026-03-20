"""
Post-process Docling (or other) PDF→Markdown to merge tables split across pages
and remove header/footer lines that appear between table fragments.

See docs/pdf_table_merge_design.md for the algorithm and rationale.
"""
from __future__ import annotations

import re
from typing import Any


def _normalize_line(line: str) -> str:
    """Collapse whitespace and strip for repetition detection."""
    return " ".join((line or "").split()).strip()


def _table_column_count(table_first_line: str) -> int:
    """Number of columns in a markdown table row (by |-delimited cells)."""
    if not table_first_line.strip():
        return 0
    # Cells are between pipes; split("|") gives ["", "cell1", "cell2", ..., ""]
    parts = [p.strip() for p in table_first_line.split("|")]
    # Drop leading/trailing empty from outer pipes
    if parts and parts[0] == "":
        parts = parts[1:]
    if parts and parts[-1] == "":
        parts = parts[:-1]
    return len(parts)


def _is_separator_row(line: str) -> bool:
    """True if line looks like |---|---| (table separator)."""
    s = line.strip()
    if not s or not s.startswith("|"):
        return False
    # Allow only |- and spaces between pipes
    return bool(re.match(r"^\|[\s\-:|]+\|$", s))


def _is_table_row(line: str) -> bool:
    """True if line looks like a markdown table row (starts with |, has content)."""
    s = line.strip()
    return s.startswith("|") and s.endswith("|") and len(s) > 2


def _parse_blocks(md: str) -> list[dict[str, Any]]:
    """
    Split markdown into blocks. Each block is either:
    - {"type": "table", "lines": [...]}  (consecutive table rows including header and separator)
    - {"type": "other", "lines": [...]}
    """
    blocks: list[dict[str, Any]] = []
    lines = md.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        if _is_table_row(line):
            table_lines = [line]
            i += 1
            while i < len(lines) and _is_table_row(lines[i]):
                table_lines.append(lines[i])
                i += 1
            blocks.append({"type": "table", "lines": table_lines})
            continue
        other_lines = [line]
        i += 1
        while i < len(lines) and not _is_table_row(lines[i]):
            other_lines.append(lines[i])
            i += 1
        blocks.append({"type": "other", "lines": other_lines})
    return blocks


def _build_repetition_map(blocks: list[dict[str, Any]]) -> dict[str, int]:
    """Count occurrences of each normalized non-table line (for header/footer detection)."""
    count: dict[str, int] = {}
    for b in blocks:
        if b["type"] == "other":
            for line in b["lines"]:
                n = _normalize_line(line)
                if len(n) >= 2 and n != "<!-- image -->":
                    count[n] = count.get(n, 0) + 1
    return count


def _is_likely_header_footer(line: str, repetition_map: dict[str, int], min_count: int = 2) -> bool:
    """True if line is repeated (candidate header/footer) or matches common patterns."""
    n = _normalize_line(line)
    if not n or n == "<!-- image -->":
        return False
    if repetition_map.get(n, 0) >= min_count:
        return True
    # Short lines that look like page numbers or noise
    if re.match(r"^[\diIvVxXlL]+$", n) and len(n) <= 4:
        return True
    # Common header/footer phrases (case-insensitive substring)
    lower = n.lower()
    if any(
        phrase in lower
        for phrase in (
            "notes to the unaudited",
            "chartered accountants",
            "independent auditor",
            "review report",
            "limited cin:",
            "associates llp",
            " llp",  # auditor/firm name (e.g. "S.R. Badiboi ... LLP,Gurugram")
        )
    ):
        return True
    return False


def _only_separator_between_tables(
    block: dict[str, Any],
    repetition_map: dict[str, int],
    min_repeat: int = 2,
) -> bool:
    """True if this 'other' block is only blank lines, images, or header/footer candidates."""
    if block["type"] != "other":
        return False
    for line in block["lines"]:
        n = _normalize_line(line)
        if not n:
            continue
        if n == "<!-- image -->":
            continue
        if _is_likely_header_footer(line, repetition_map, min_count=min_repeat):
            continue
        return False
    return True


def _table_body_rows(lines: list[str]) -> list[str]:
    """Table lines without the first row (header) and without the separator row (|---|---|)."""
    if not lines:
        return []
    out: list[str] = []
    for i, line in enumerate(lines):
        if i == 0:
            continue  # skip header
        if _is_separator_row(line):
            continue
        out.append(line)
    return out


def _table_data_rows_only(lines: list[str]) -> list[str]:
    """All table rows that are not separator rows (for continuation table: no header to skip)."""
    return [line for line in lines if not _is_separator_row(line)]


def merge_split_tables_and_remove_header_footer(
    markdown: str,
    min_repetition_for_header_footer: int = 2,
) -> str:
    """
    Post-process markdown: remove header/footer lines that sit between tables,
    and merge consecutive tables that have the same column count (same logical table).

    Algorithm (see docs/pdf_table_merge_design.md):
    1. Parse into table blocks and other blocks.
    2. Build repetition map; mark repeated (or pattern-matched) lines as header/footer.
    3. Between two table blocks: if the only content is blank/image/header-footer, treat as separator.
    4. Merge consecutive tables that are separated only by separator and have same column count.

    Args:
        markdown: Full markdown string (e.g. from Docling).
        min_repetition_for_header_footer: Min occurrences for a line to be treated as header/footer (default 2).

    Returns:
        Cleaned markdown with merged tables and separator blocks between them removed.
    """
    if not (markdown or "").strip():
        return markdown

    blocks = _parse_blocks(markdown)
    repetition_map = _build_repetition_map(blocks)

    # Merge adjacent tables when only separator between them and same column count
    result_blocks: list[dict[str, Any]] = []
    i = 0
    while i < len(blocks):
        current = blocks[i]
        if current["type"] != "table":
            result_blocks.append(current)
            i += 1
            continue
        # Collect consecutive tables separated only by separator blocks
        merged_lines = list(current["lines"])
        merged_cols = _table_column_count(merged_lines[0]) if merged_lines else 0
        i += 1
        while i < len(blocks):
            # Skip separator-only "other" blocks
            if blocks[i]["type"] == "other" and _only_separator_between_tables(
                blocks[i], repetition_map, min_repeat=min_repetition_for_header_footer
            ):
                i += 1
                continue
            if blocks[i]["type"] != "table":
                break
            next_table = blocks[i]
            next_lines = next_table["lines"]
            next_cols = _table_column_count(next_lines[0]) if next_lines else 0
            if next_cols != merged_cols or merged_cols == 0:
                break
            # Merge: keep first table header + separator, then first body, then next table data rows (all, no header skip)
            body1 = _table_body_rows(merged_lines)
            body2 = _table_data_rows_only(next_lines)
            # First table: header row + separator row (if any) + body
            header_and_sep: list[str] = []
            for line in merged_lines:
                header_and_sep.append(line)
                if _is_separator_row(line):
                    break
            merged_lines = header_and_sep + body1 + body2
            merged_cols = next_cols
            i += 1
        result_blocks.append({"type": "table", "lines": merged_lines})
    # Rebuild markdown
    out_lines: list[str] = []
    for b in result_blocks:
        lines = b["lines"]
        if b["type"] == "table":
            out_lines.extend(lines)
        else:
            # Emit "other" block only if not a separator between tables (we already consumed those)
            # Here we only have non-separator "other" blocks since we skipped separator in the loop
            out_lines.extend(lines)
        out_lines.append("")  # blank between blocks
    return "\n".join(out_lines).strip() + "\n"

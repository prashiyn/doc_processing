from __future__ import annotations

"""
End-to-end Financial Filings Pipeline (FFP) for chunk generation.

Entrypoint: `FinancialFilingsPipeline.run`.

The final output is a list of chunk objects consistent with
`docs/DOCUMENT_CHUNKING.md` (see "Chunk object" schema).
"""

from dataclasses import dataclass
from pathlib import Path
import shutil
import tempfile
from typing import Any, Iterable, Literal

from doc_processing.ffp.multimodal.table_extractor import TableExtractor
from doc_processing.ffp.multimodal.image_captioner import ImageCaptioner
from doc_processing.ffp.processing.chunk_segmenter import ChunkSegmenter
from doc_processing.ffp.processing.deduplicator import Deduplicator
from doc_processing.ffp.processing.coreference_resolver import CoreferenceResolver
from doc_processing.ffp.processing.section_summarizer import SectionSummarizer


Block = dict[str, Any]


@dataclass
class Chunk:
    """Canonical chunk object: content (original text/table/image), type, and metadata."""

    chunk_id: str
    content: str
    type: str  # "text" | "table" | "image"
    doc_id: str
    page: int | None
    bundle_id: str
    section_title: str | None
    title_summary: str
    publish_date: str | None
    prev_chunk: str | None
    next_chunk: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id,
            "content": self.content,
            "type": self.type,
            "doc_id": self.doc_id,
            "page": self.page,
            "bundle_id": self.bundle_id,
            "section_title": self.section_title,
            "title_summary": self.title_summary,
            "publish_date": self.publish_date,
            "prev_chunk": self.prev_chunk,
            "next_chunk": self.next_chunk,
        }


class FinancialFilingsPipeline:
    """High-level orchestrator for document chunking."""

    def __init__(
        self,
        *,
        max_chunk_chars: int = 900,
        min_block_chars: int = 200,
        dedup_threshold: float | None = None,
    ) -> None:
        self._segmenter = ChunkSegmenter(
            max_chars=max_chunk_chars,
            min_block_chars=min_block_chars,
        )
        self._deduplicator = Deduplicator(threshold=dedup_threshold)
        self._coref = CoreferenceResolver()
        self._summarizer = SectionSummarizer()
        self._table_extractor = TableExtractor()
        self._image_captioner = ImageCaptioner()

    # ---------- Public API ----------

    def run(
        self,
        source: str | Path | bytes,
        *,
        doc_id: str,
        publish_date: str | None = None,
        file_extension: str | None = None,
        pdf_backend: Literal["easyocr", "vlm"] = "vlm",
    ) -> list[dict[str, Any]]:
        """
        Parse a document with Docling (PDF/Markdown/HTML), then run chunking pipeline.

        For `bytes` input, `file_extension` is required (e.g. ".pdf", ".md", ".html").
        """
        conv_result = self._parse_with_docling(
            source,
            file_extension=file_extension,
            pdf_backend=pdf_backend,
        )
        if conv_result is None:
            return []
        return self._run_docling_result(
            conv_result,
            doc_id=doc_id,
            publish_date=publish_date,
        )

    def run_from_pdf(
        self,
        source: str | Path | bytes,
        *,
        doc_id: str,
        publish_date: str | None = None,
        backend: Literal["easyocr", "vlm"] = "vlm",
    ) -> list[dict[str, Any]]:
        """Convenience wrapper for PDF input."""
        return self.run(
            source,
            doc_id=doc_id,
            publish_date=publish_date,
            file_extension=".pdf" if isinstance(source, bytes) else None,
            pdf_backend=backend,
        )

    def run_from_markdown(
        self,
        markdown: str | Path,
        *,
        doc_id: str,
        publish_date: str | None = None,
    ) -> list[dict[str, Any]]:
        """Convenience wrapper for Markdown input."""
        ext = ".md" if isinstance(markdown, bytes) else None
        return self.run(markdown, doc_id=doc_id, publish_date=publish_date, file_extension=ext)

    def run_from_html(
        self,
        html: str | Path | bytes,
        *,
        doc_id: str,
        publish_date: str | None = None,
        file_extension: str | None = None,
    ) -> list[dict[str, Any]]:
        """Convenience wrapper for HTML input."""
        return self.run(
            html,
            doc_id=doc_id,
            publish_date=publish_date,
            file_extension=file_extension,
        )

    # ---------- Internal steps ----------

    def _parse_with_docling(
        self,
        source: str | Path | bytes,
        *,
        file_extension: str | None,
        pdf_backend: Literal["easyocr", "vlm"] = "vlm",
    ) -> Any:
        from doc_processing.services.docling_parser import (
            parse_html_using_docling,
            parse_markdown_using_docling,
            parse_pdf_using_docling,
        )

        ext = self._infer_extension(source, file_extension)
        if ext == ".pdf":
            return parse_pdf_using_docling(
                source,
                file_extension=file_extension,
                backend=pdf_backend,
                embed_image=True,
            )
        if ext in (".md", ".markdown"):
            return parse_markdown_using_docling(
                source,
                file_extension=file_extension,
                embed_image=True,
            )
        if ext in (".html", ".htm", ".xhtml"):
            return parse_html_using_docling(
                source,
                file_extension=file_extension,
                embed_image=True,
            )
        return None

    @staticmethod
    def _infer_extension(source: str | Path | bytes, file_extension: str | None) -> str | None:
        if isinstance(source, bytes):
            if not file_extension:
                return None
            ext = file_extension.strip().lower()
            return ext if ext.startswith(".") else f".{ext}"
        try:
            return Path(source).suffix.lower()  # type: ignore[arg-type]
        except Exception:
            return None

    def _run_docling_result(
        self,
        conv_result: Any,
        *,
        doc_id: str,
        publish_date: str | None,
    ) -> list[dict[str, Any]]:
        temp_dir = Path(tempfile.mkdtemp(prefix="ffp_docling_"))
        try:
            blocks = self._docling_to_blocks(conv_result, temp_dir=temp_dir)
            enriched_blocks = self._enrich_multimodal_blocks(blocks)
            return self._run_blocks(enriched_blocks, doc_id=doc_id, publish_date=publish_date)
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _docling_to_blocks(self, conv_result: Any, *, temp_dir: Path) -> list[Block]:
        """Convert a Docling conversion result into normalized pipeline blocks."""
        doc = getattr(conv_result, "document", None)
        if doc is None or not hasattr(doc, "iterate_items"):
            return []

        blocks: list[Block] = []
        for element, level in doc.iterate_items():
            elem_type = self._docling_item_type(element)
            text = self._docling_item_text(doc, element, elem_type)
            page = self._docling_item_page(element)
            img_path = self._docling_item_image_path(doc, element, elem_type, temp_dir=temp_dir)
            lvl: int | None = None
            try:
                lvl = int(level) + 1 if level is not None else None
            except Exception:
                lvl = None

            blocks.append(
                {
                    "type": elem_type,
                    "text": text,
                    "level": lvl,
                    "page": page,
                    "img_path": img_path,
                }
            )
        return blocks

    @staticmethod
    def _docling_item_type(element: Any) -> str:
        name = element.__class__.__name__.lower()
        if "table" in name:
            return "table"
        if "picture" in name or "image" in name:
            return "image"
        return "text"

    @staticmethod
    def _docling_item_text(doc: Any, element: Any, elem_type: str) -> str | None:
        if elem_type in ("table", "text"):
            for attr in ("text", "orig", "raw_text", "content"):
                val = getattr(element, attr, None)
                if isinstance(val, str) and val.strip():
                    return val.strip()
            if hasattr(element, "export_to_markdown"):
                try:
                    md = element.export_to_markdown(doc)
                    if isinstance(md, str) and md.strip():
                        return md.strip()
                except Exception:
                    pass
        if elem_type == "image":
            # Prefer Docling's generated caption (from picture description),
            # if available. In Docling, `caption_text` is commonly a method:
            #   element.caption_text(doc=document)
            caption_fn = getattr(element, "caption_text", None)
            if callable(caption_fn):
                try:
                    cap = caption_fn(doc=doc)
                except TypeError:
                    cap = caption_fn(doc)
                if isinstance(cap, str) and cap.strip():
                    return cap.strip()

            # Fallback: prefer any textual caption/description stored directly.
            for attr in ("text", "caption", "description", "content"):
                val = getattr(element, attr, None)
                if isinstance(val, str) and val.strip():
                    return val.strip()
        return None

    @staticmethod
    def _docling_item_page(element: Any) -> int | None:
        prov = getattr(element, "prov", None)
        if not prov:
            return None
        try:
            page_no = getattr(prov[0], "page_no", None)
            return int(page_no) if page_no is not None else None
        except Exception:
            return None

    @staticmethod
    def _docling_item_image_path(
        doc: Any,
        element: Any,
        elem_type: str,
        *,
        temp_dir: Path,
    ) -> str | None:
        if elem_type not in ("table", "image") or not hasattr(element, "get_image"):
            return None
        try:
            img = element.get_image(doc)
            if img is None:
                return None
            out = temp_dir / f"{elem_type}_{len(list(temp_dir.glob('*'))):06d}.png"
            img.save(str(out), "PNG")
            return str(out)
        except Exception:
            return None

    def _enrich_multimodal_blocks(self, blocks: Iterable[Block]) -> list[Block]:
        """Convert `table` and `image` blocks into textual narratives."""
        enriched: list[Block] = []
        for b in blocks:
            b_type = (b.get("type") or "text").lower()
            if b_type == "table" and not b.get("text") and b.get("img_path"):
                summary = self._table_extractor.convert(b["img_path"])
                nb = dict(b)
                nb["text"] = summary
                enriched.append(nb)
            elif b_type == "image" and not b.get("text") and b.get("img_path"):
                caption = self._image_captioner.caption(b["img_path"])
                nb = dict(b)
                nb["text"] = caption
                enriched.append(nb)
            else:
                enriched.append(b)
        return enriched

    def _run_blocks(
        self,
        blocks: Iterable[Block],
        *,
        doc_id: str,
        publish_date: str | None,
    ) -> list[dict[str, Any]]:
        # 1. Segment into bundle-aware chunks (bundle_id, section_title, content, type)
        chunks = self._segmenter.segment(blocks)

        # 2. Deduplicate by content similarity
        chunks = self._deduplicator.run(chunks)

        # 3. Coreference resolution
        chunks = self._coref.resolve(chunks)

        # 4. Section-level summaries
        chunks = self._summarizer.summarize(chunks)

        # 5. Attach IDs, doc-level metadata, and prev/next pointers
        return self._finalize_chunks(chunks, doc_id=doc_id, publish_date=publish_date)

    def _finalize_chunks(
        self,
        chunks: Iterable[dict[str, Any]],
        *,
        doc_id: str,
        publish_date: str | None,
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        prev_id: str | None = None

        for idx, c in enumerate(chunks, start=1):
            chunk_id = f"{doc_id}_{idx:05d}"
            content = c.get("content") or ""
            chunk_type = c.get("type") or "text"
            page = c.get("page")
            bundle_id = c.get("bundle_id") or ""
            section_title = c.get("section_title")
            title_summary = c.get("title_summary") or ""

            next_id: str | None = None  # filled in next iteration
            # We'll set next_id of previous chunk once we know current id.
            if prev_id is not None and out:
                out[-1]["next_chunk"] = chunk_id
                out[-1]["prev_chunk"] = out[-1]["prev_chunk"]  # keep existing

            chunk = Chunk(
                chunk_id=chunk_id,
                content=content,
                type=chunk_type,
                doc_id=doc_id,
                page=page,
                bundle_id=bundle_id,
                section_title=section_title,
                title_summary=title_summary,
                publish_date=publish_date,
                prev_chunk=prev_id,
                next_chunk=next_id,
            )
            out.append(chunk.to_dict())
            prev_id = chunk_id

        # Fix next_chunk for the last element (already None by construction).
        return out


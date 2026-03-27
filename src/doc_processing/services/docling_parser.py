"""Docling-based parsers for PDF/XBRL and markdown export helpers."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import shutil
import tempfile

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    EasyOcrOptions,
    PdfPipelineOptions,
    PictureDescriptionVlmEngineOptions,
    TableStructureOptions,
    VlmConvertOptions,
    VlmPipelineOptions,
)
from docling.datamodel.vlm_engine_options import (
    ApiVlmEngineOptions,
    VlmEngineType,
    MlxVlmEngineOptions
)
from docling.document_converter import DocumentConverter, PdfFormatOption, XBRLFormatOption
from docling.pipeline.vlm_pipeline import VlmPipeline


def _configure_picture_options(pipeline_options: Any) -> None:
    """Enable picture handling options when available on a pipeline options object."""
    if hasattr(pipeline_options, "generate_picture_images"):
        pipeline_options.generate_picture_images = True
    if hasattr(pipeline_options, "do_picture_description"):
        pipeline_options.do_picture_description = True
    if hasattr(pipeline_options, "do_picture_classification"):
        pipeline_options.do_picture_classification = True
    pdesc = getattr(pipeline_options, "picture_description_options", None)
    if pdesc is not None and hasattr(pdesc, "picture_area_threshold"):
        pdesc.picture_area_threshold = 0.05


def _configure_picture_description_vlm(pipeline_options: Any) -> None:
    """Configure Docling picture-description VLM options consistently."""
    try:
        pipeline_options.enable_remote_services = True
    except Exception:
        pass

    try:
        pipeline_options.picture_description_options = (
            PictureDescriptionVlmEngineOptions.from_preset(
                "granite_vision",
                engine_options=MlxVlmEngineOptions(
                    engine_type=VlmEngineType.API_OLLAMA,
                ),
            )
        )
        pipeline_options.picture_description_options.prompt = (
            "Describe the image in three sentences. Be concise and accurate."
        )
        pipeline_options.picture_description_options.picture_area_threshold = 0.05
    except Exception:
        pass


def compare_images(
    hashes: list[Any],
    *,
    max_hamming_distance: int = 5,
) -> set[int]:
    """Identify visually similar images by comparing perceptual hashes.

    Keeps the first occurrence of each unique image and returns indices to remove.
    """
    keep_indices: list[int] = []
    remove_indices: set[int] = set()

    for i, h in enumerate(hashes):
        is_duplicate = False
        for j in keep_indices:
            # `imagehash.ImageHash` implements hamming distance via subtraction.
            try:
                dist = h - hashes[j]
            except Exception:
                dist = 999999
            if dist <= max_hamming_distance:
                is_duplicate = True
                break
        if is_duplicate:
            remove_indices.add(i)
        else:
            keep_indices.append(i)

    return remove_indices


def _dedupe_common_pictures_in_doc(
    doc: Any,
    *,
    max_hamming_distance: int = 5,
) -> None:
    """Remove repeated pictures (logos) from a DoclingDocument, preserving order.

    Strategy:
    - Iterate pictures in document order
    - Hash each picture image with perceptual hashing
    - Delete any picture whose hash is similar to a previously-kept picture
    """
    try:
        import imagehash  # type: ignore
    except Exception:
        # If imagehash isn't available, skip deduplication rather than failing parsing.
        return

    try:
        from PIL import Image  # noqa: F401
    except Exception:
        return

    pictures: list[Any] = []
    hashes: list[Any] = []

    # Iterate pictures in document order. traverse_pictures helps find nested picture nodes.
    for element, _level in doc.iterate_items(traverse_pictures=True):
        name = element.__class__.__name__.lower()
        if "picture" not in name:
            continue
        try:
            img = element.get_image(doc)
            if img is None:
                continue
            # Normalize mode for consistent hashing.
            img = img.convert("RGB")
            h = imagehash.phash(img)
        except Exception:
            continue
        pictures.append(element)
        hashes.append(h)

    if not pictures or not hashes:
        return

    remove_indices = compare_images(hashes, max_hamming_distance=max_hamming_distance)
    if not remove_indices:
        return

    node_items_to_delete = [pictures[i] for i in sorted(remove_indices)]
    try:
        doc.delete_items(node_items=node_items_to_delete)
    except TypeError:
        # Older Docling versions may use a positional argument or different keyword.
        try:
            doc.delete_items(node_items_to_delete)
        except Exception:
            return
    except Exception:
        return


def _prepare_local_source_path(
    source: str | Path | bytes,
    *,
    expected_exts: tuple[str, ...],
    file_extension: str | None = None,
) -> tuple[Path | None, bool]:
    """Return (path, should_cleanup) with validation for local inputs."""
    if isinstance(source, bytes):
        if not file_extension:
            return None, False
        ext = file_extension.strip().lower()
        if not ext.startswith("."):
            ext = f".{ext}"
        if ext not in expected_exts:
            return None, False
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
            f.write(source)
            return Path(f.name), True

    path = Path(source)
    if not path.exists() or path.suffix.lower() not in expected_exts:
        return None, False
    return path, False


def parse_pdf_using_docling(
    source: str | Path | bytes,
    file_extension: str | None = None,
    backend: Literal["easyocr", "vlm"] = "easyocr",
    embed_image: bool = True,
) -> Any:
    """
    Parse a PDF with Docling and return the Docling conversion result object.

    This function performs the heavy lifting for PDF parsing and pipeline setup.
    Callers can then serialize the returned `conv_result.document` as needed.

    Returns:
        Docling conversion result on success; None on failure/invalid input.
    """
    path: Path | None = None
    try:
        if isinstance(source, bytes):
            if not file_extension or ".pdf" not in file_extension.strip().lower():
                return None
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as f:
                f.write(source)
                path = Path(f.name)
        else:
            path = Path(source)

        if not path or not path.exists() or path.suffix.lower() != ".pdf":
            return None

        if backend == "easyocr":
            pipeline_options = PdfPipelineOptions()
            pipeline_options.do_ocr = True
            pipeline_options.ocr_options = EasyOcrOptions()
            pipeline_options.do_table_structure = True
            pipeline_options.table_structure_options = TableStructureOptions(
                do_cell_matching=True
            )
            if embed_image:
                _configure_picture_options(pipeline_options)
                _configure_picture_description_vlm(pipeline_options)
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)
                }
            )
        else:
            vlm_options = VlmConvertOptions.from_preset(
                "granite_docling",
                engine_options=ApiVlmEngineOptions(
                    runtime_type=VlmEngineType.API_OLLAMA,
                    timeout=90,
                    params={"model": "ibm/granite-docling:latest"},
                ),
            )
            pipeline_options = VlmPipelineOptions(
                vlm_options=vlm_options,
                enable_remote_services=True,
            )
            if embed_image:
                _configure_picture_options(pipeline_options)
                _configure_picture_description_vlm(pipeline_options)
            doc_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pipeline_options,
                        pipeline_cls=VlmPipeline,
                    )
                }
            )

        conv_result = doc_converter.convert(str(path))
        try:
            doc = getattr(conv_result, "document", None)
            if doc is not None:
                _dedupe_common_pictures_in_doc(doc)
        except Exception:
            pass
        return conv_result
    except Exception:
        return None
    finally:
        if isinstance(source, bytes) and path is not None:
            path.unlink(missing_ok=True)


def parse_markdown_using_docling(
    source: str | Path | bytes,
    file_extension: str | None = None,
    embed_image: bool = True,
) -> Any:
    """
    Parse a Markdown document with Docling and return the conversion result object.

    Markdown is already structured, so no VLM branch is used.
    """
    path, should_cleanup = _prepare_local_source_path(
        source,
        expected_exts=(".md", ".markdown"),
        file_extension=file_extension,
    )
    if path is None:
        return None

    try:
        converter = DocumentConverter(allowed_formats=[InputFormat.MD])
        if embed_image:
            option = converter.format_to_options.get(InputFormat.MD)
            if option is not None:
                pipeline_options = getattr(option, "pipeline_options", None)
                if pipeline_options is not None:
                    _configure_picture_options(pipeline_options)
                    _configure_picture_description_vlm(pipeline_options)

        conv_result = converter.convert(str(path))
        try:
            doc = getattr(conv_result, "document", None)
            if doc is not None:
                _dedupe_common_pictures_in_doc(doc)
        except Exception:
            pass
        return conv_result
    except Exception:
        return None
    finally:
        if should_cleanup and path is not None:
            path.unlink(missing_ok=True)


def parse_html_using_docling(
    source: str | Path | bytes,
    file_extension: str | None = None,
    embed_image: bool = True,
) -> Any:
    """
    Parse an HTML/XHTML document with Docling and return the conversion result object.

    HTML is already structured, so no VLM branch is used.
    """
    path, should_cleanup = _prepare_local_source_path(
        source,
        expected_exts=(".html", ".htm", ".xhtml"),
        file_extension=file_extension,
    )
    if path is None:
        return None

    try:
        converter = DocumentConverter(allowed_formats=[InputFormat.HTML, InputFormat.XHTML])
        if embed_image:
            for fmt in (InputFormat.HTML, InputFormat.XHTML):
                option = converter.format_to_options.get(fmt)
                if option is None:
                    continue
                pipeline_options = getattr(option, "pipeline_options", None)
                if pipeline_options is not None:
                    _configure_picture_options(pipeline_options)
                    _configure_picture_description_vlm(pipeline_options)
        conv_result = converter.convert(str(path))
        try:
            doc = getattr(conv_result, "document", None)
            if doc is not None:
                _dedupe_common_pictures_in_doc(doc)
        except Exception:
            pass
        return conv_result
    except Exception:
        return None
    finally:
        if should_cleanup and path is not None:
            path.unlink(missing_ok=True)


def pdf_to_markdown_docling(
    source: str | Path | bytes,
    file_extension: str | None = None,
    backend: Literal["easyocr", "vlm"] = "easyocr",
    merge_tables: bool = False,
    embed_image: bool = True,
) -> tuple[str, list[Any]]:
    """
    Convert a PDF to markdown using Docling.

    Signature intentionally kept stable for existing callers.
    """
    conv_result = parse_pdf_using_docling(
        source=source,
        file_extension=file_extension,
        backend=backend,
        embed_image=embed_image,
    )
    if conv_result is None:
        return ("", [])

    try:
        if embed_image:
            from docling_core.types.doc import ImageRefMode

            tmp_md: Path | None = None
            try:
                tmp_md = Path(tempfile.mkdtemp(prefix="docling_md_")) / "doc.md"
                conv_result.document.save_as_markdown(tmp_md, image_mode=ImageRefMode.EMBEDDED)
                markdown = (tmp_md.read_text(encoding="utf-8") or "").strip()
            finally:
                if tmp_md is not None:
                    try:
                        shutil.rmtree(tmp_md.parent, ignore_errors=True)
                    except OSError:
                        pass
        else:
            markdown = (conv_result.document.export_to_markdown() or "").strip()

        if merge_tables:
            from doc_processing.services.pdf_markdown_cleanup import (
                merge_split_tables_and_remove_header_footer,
            )

            markdown = merge_split_tables_and_remove_header_footer(markdown)
        return (markdown, [])
    except Exception:
        return ("", [])


def xbrl_to_markdown(
    source: str | Path | bytes,
    file_extension: str | None = None,
    taxonomy_dir: Path | None = None,
) -> str:
    """Convert an XBRL instance document to markdown using Docling's XBRL backend."""
    from doc_processing.config import get_settings
    from docling.datamodel.backend_options import XBRLBackendOptions

    path: Path | None = None
    if isinstance(source, bytes):
        if not file_extension or all(ext not in file_extension.lower() for ext in (".xml", ".xbrl")):
            return ""
        with tempfile.NamedTemporaryFile(suffix=file_extension or ".xml", delete=False) as f:
            f.write(source)
            path = Path(f.name)
    else:
        path = Path(source)

    if not path or not path.exists():
        return ""
    if path.suffix.lower() not in (".xml", ".xbrl"):
        return ""

    settings = get_settings()
    if taxonomy_dir is None:
        if getattr(settings, "xbrl_taxonomy_dir", None):
            taxonomy_dir = Path(settings.xbrl_taxonomy_dir).expanduser().resolve()
        else:
            taxonomy_dir = Path("data") / "nse"

    try:
        backend_options = XBRLBackendOptions(
            enable_local_fetch=True,
            enable_remote_fetch=True,
            taxonomy=taxonomy_dir,
        )
        converter = DocumentConverter(
            allowed_formats=[InputFormat.XML_XBRL],
            format_options={
                InputFormat.XML_XBRL: XBRLFormatOption(backend_options=backend_options)
            },
        )
        result = converter.convert(str(path))
        doc = result.document
        markdown = (doc.export_to_markdown() or "").strip()
        return markdown
    except Exception:
        return ""
    finally:
        if isinstance(source, bytes) and path is not None:
            path.unlink(missing_ok=True)


"""Document processing and chunking endpoints.

All document-processing endpoints (except `/chunk`) support *either*:
- **URL input**: the file is fetched with NSE-aware logic and written to a temp directory, then deleted.
- **File upload**: the file bytes are processed directly.
"""

import asyncio
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, HttpUrl

from doc_processing.services import (
    chunk_text,
    file_to_markdown_using_markitdown,
    ocr_to_markdown_deepseek_image,
    ocr_to_markdown_deepseek_pdf,
    ocr_to_markdown_glm,
    pdf_to_markdown_docling,
    xbrl_to_markdown,
    process_file,
)
from doc_processing.services.document_fetch import (
    DocumentFetchError,
    fetch_document,
    temp_path_for_document,
)
from doc_processing.services.file_processor import FileProcessorResult

router = APIRouter()


class ChunkRequest(BaseModel):
    """Request body for text chunking."""
    text: str = Field(..., description="Raw text to split into chunks")
    chunk_size: int = Field(512, ge=1, le=8192, description="Target size per chunk in characters")
    chunk_overlap: int = Field(64, ge=0, le=2048, description="Overlap between consecutive chunks")


class ChunkResponse(BaseModel):
    """List of text chunks for RAG."""
    chunks: list[str] = Field(..., description="Text chunks")


class ConvertResponse(BaseModel):
    """Result of document-to-markdown conversion."""
    markdown: str = Field(..., description="Converted markdown content")


class TableItem(BaseModel):
    """One extracted table."""
    text: str = Field(..., description="Plain-text table content")
    text_as_html: str | None = Field(None, description="Table as HTML when available")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Element metadata")


class ImageItem(BaseModel):
    """One extracted image block."""
    metadata: dict[str, Any] = Field(default_factory=dict, description="Element metadata")
    image_base64: str | None = Field(None, description="Base64-encoded image when in payload")


class ProcessFromUrlResponse(BaseModel):
    """Result of process-from-url: tables and images."""
    tables: list[TableItem] = Field(..., description="Extracted tables")
    images: list[ImageItem] = Field(..., description="Extracted images")
    extract_output_dir: str | None = Field(None, description="Dir path if images saved to disk")


class PdfToMarkdownResponse(BaseModel):
    """Result of PDF-to-markdown conversion."""
    markdown: str = Field(..., description="Converted markdown content")


class XbrlToMarkdownResponse(BaseModel):
    """Result of XBRL-to-markdown conversion."""
    markdown: str = Field(..., description="Converted markdown content")


def _infer_file_type_from_url(url: str, content_type: str | None) -> Literal["pdf", "image", "markdown"]:
    path = url.split("?")[0].lower()
    if path.endswith(".pdf"):
        return "pdf"
    if path.endswith((".md", ".markdown")):
        return "markdown"
    if any(path.endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif", ".gif")):
        return "image"
    if content_type:
        if "pdf" in content_type:
            return "pdf"
        if "markdown" in content_type or "text/markdown" in content_type:
            return "markdown"
        if "image/" in content_type:
            return "image"
    return "pdf"


def _suffix_from_url(url: str, default: str = ".bin") -> str:
    path = url.split("?")[0].lower()
    if "." in path:
        return "." + path.rsplit(".", 1)[-1]
    return default


async def _source_from_url_to_temp(url: str, timeout: int, *, default_suffix: str = ".bin") -> Path:
    """
    Fetch a URL and write it to a temp file. Always returns a local path.
    Caller must delete the path after processing.
    """
    content = await asyncio.to_thread(fetch_document, url, timeout)
    suffix = _suffix_from_url(url, default=default_suffix)
    return temp_path_for_document(content, suffix=suffix)


async def _read_upload(file: UploadFile) -> tuple[bytes, str | None]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Empty file")
    return content, _infer_extension_from_upload(file)


def _serialize_process_result(result: FileProcessorResult) -> ProcessFromUrlResponse:
    return ProcessFromUrlResponse(
        tables=[
            TableItem(text=t.text, text_as_html=t.text_as_html, metadata=t.metadata)
            for t in result.tables
        ],
        images=[
            ImageItem(metadata=i.metadata, image_base64=i.image_base64)
            for i in result.images
        ],
        extract_output_dir=str(result.extract_output_dir) if result.extract_output_dir else None,
    )


def _http_exception_for_fetch_error(e: DocumentFetchError) -> HTTPException:
    """Map DocumentFetchError to appropriate HTTP status."""
    if "timed out" in str(e).lower():
        return HTTPException(status_code=504, detail=str(e))
    if e.status_code is not None and 400 <= e.status_code < 500:
        return HTTPException(status_code=422, detail=str(e))
    if e.status_code is not None and e.status_code >= 500:
        return HTTPException(status_code=502, detail=f"Upstream error: {e}")
    return HTTPException(status_code=422, detail=str(e))


_MS_DOC_EXTENSIONS: set[str] = {
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
}

_MS_DOC_MIME_TYPES: set[str] = {
    # Word
    "application/msword",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    # Excel
    "application/vnd.ms-excel",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    # PowerPoint
    "application/vnd.ms-powerpoint",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def _infer_extension_from_upload(file: UploadFile) -> str | None:
    fn = (file.filename or "").strip()
    if "." in fn:
        return f".{fn.rsplit('.', 1)[-1]}".lower()
    ct = (file.content_type or "").lower()
    if ct == "application/msword":
        return ".doc"
    if ct == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        return ".docx"
    if ct == "application/vnd.ms-excel":
        return ".xls"
    if ct == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
        return ".xlsx"
    if ct == "application/vnd.ms-powerpoint":
        return ".ppt"
    if ct == "application/vnd.openxmlformats-officedocument.presentationml.presentation":
        return ".pptx"
    return None


_IMAGE_EXTENSIONS: set[str] = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".bmp",
    ".tiff",
    ".tif",
}


@router.post("/convert-ms-docs", response_model=ConvertResponse)
async def convert_ms_docs(
    file: UploadFile | None = File(None),
    url: HttpUrl | None = None,
) -> ConvertResponse:
    """Convert a Microsoft Office document (Word/Excel/PowerPoint) to markdown from file upload or URL."""
    try:
        if file is not None:
            content, ext = await _read_upload(file)
            ct = (file.content_type or "").lower()
        elif url is not None:
            # For URL: download to temp file and infer extension from URL
            tmp: Path | None = None
            try:
                tmp = await _source_from_url_to_temp(str(url), 60)
                ext = tmp.suffix.lower() or _suffix_from_url(str(url), default=".bin")
                ct = ""
                if ext not in _MS_DOC_EXTENSIONS:
                    raise HTTPException(
                        status_code=415,
                        detail=(
                            "Unsupported document type. This endpoint accepts only Microsoft Office files: "
                            ".doc/.docx/.xls/.xlsx/.ppt/.pptx"
                        ),
                    )
                md = await asyncio.to_thread(file_to_markdown_using_markitdown, tmp)
                return ConvertResponse(markdown=md)
            finally:
                if tmp is not None:
                    tmp.unlink(missing_ok=True)
        else:
            raise HTTPException(status_code=422, detail="Either file or url must be provided")

        if (ext not in _MS_DOC_EXTENSIONS) and (ct not in _MS_DOC_MIME_TYPES):
            raise HTTPException(
                status_code=415,
                detail=(
                    "Unsupported document type. This endpoint accepts only Microsoft Office files: "
                    ".doc/.docx/.xls/.xlsx/.ppt/.pptx"
                ),
            )

        md = file_to_markdown_using_markitdown(content, file_extension=ext)
        return ConvertResponse(markdown=md)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e


@router.post("/process-using-unstructured", response_model=ProcessFromUrlResponse)
async def process_using_unstructured(
    file: UploadFile | None = File(None),
    url: HttpUrl | None = None,
    file_type: Literal["pdf", "image", "markdown"] | None = None,
) -> ProcessFromUrlResponse:
    """
    Process a PDF or image using Unstructured: extract tables, images, and metadata.

    Accepts either an uploaded file or a URL.
    """
    tmp: Path | None = None
    try:
        if file is not None:
            content, _ = await _read_upload(file)
            inferred_type = file_type or _infer_file_type_from_url(file.filename or "", None)
            result = await asyncio.to_thread(
                process_file,
                content,
                file_type=inferred_type,
            )
            return _serialize_process_result(result)

        if url is not None:
            tmp = await _source_from_url_to_temp(str(url), 60)
            inferred_type = file_type or _infer_file_type_from_url(str(url), None)
            result = await asyncio.to_thread(
                process_file,
                tmp,
                file_type=inferred_type,
            )
            return _serialize_process_result(result)

        raise HTTPException(status_code=422, detail="Either file or url must be provided")
    except DocumentFetchError as e:
        raise _http_exception_for_fetch_error(e)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


@router.post("/pdf-to-markdown", response_model=PdfToMarkdownResponse)
async def pdf_to_markdown_endpoint(
    file: UploadFile | None = File(None),
    url: HttpUrl | None = None,
    backend: Literal["easyocr", "vlm"] = "easyocr",
    merge_tables: bool = False,
) -> PdfToMarkdownResponse:
    """Convert a PDF to markdown using Docling (easyocr or vlm backend) from file upload or URL."""
    tmp: Path | None = None
    try:
        if file is not None:
            content, ext = await _read_upload(file)
            md, _ = await asyncio.to_thread(
                pdf_to_markdown_docling,
                content,
                file_extension=ext or ".pdf",
                backend=backend,
                merge_tables=merge_tables,
            )
            return PdfToMarkdownResponse(markdown=md)

        if url is not None:
            tmp = await _source_from_url_to_temp(str(url), 120, default_suffix=".pdf")
            md, _ = await asyncio.to_thread(
                pdf_to_markdown_docling,
                tmp,
                backend=backend,
                merge_tables=merge_tables,
            )
            return PdfToMarkdownResponse(markdown=md)

        raise HTTPException(status_code=422, detail="Either file or url must be provided")
    except DocumentFetchError as e:
        raise _http_exception_for_fetch_error(e)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


@router.post("/xbrl-to-markdown", response_model=XbrlToMarkdownResponse)
async def xbrl_to_markdown_endpoint(
    file: UploadFile | None = File(None),
    url: HttpUrl | None = None,
) -> XbrlToMarkdownResponse:
    """Convert an XBRL instance document (.xml/.xbrl) to markdown using Docling's XBRL backend from file or URL."""
    tmp: Path | None = None
    try:
        if file is not None:
            content, ext = await _read_upload(file)
            md = await asyncio.to_thread(xbrl_to_markdown, content, ext or ".xml")
            return XbrlToMarkdownResponse(markdown=md)

        if url is not None:
            tmp = await _source_from_url_to_temp(str(url), 120, default_suffix=".xml")
            md = await asyncio.to_thread(xbrl_to_markdown, tmp)
            return XbrlToMarkdownResponse(markdown=md)

        raise HTTPException(status_code=422, detail="Either file or url must be provided")
    except DocumentFetchError as e:
        raise _http_exception_for_fetch_error(e)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


@router.post("/process-using-deepseek", response_model=PdfToMarkdownResponse)
async def process_using_deepseek(
    file: UploadFile | None = File(None),
    url: HttpUrl | None = None,
) -> PdfToMarkdownResponse:
    """
    Convert a PDF or image to markdown using DeepSeek-OCR via Ollama.

    Accepts either an uploaded file or a URL. Only PDF and common image types are supported.
    """
    tmp: Path | None = None
    try:
        # File upload path: use bytes and file_extension
        if file is not None:
            content, ext = await _read_upload(file)
            ext = (ext or "").lower()
            if ext == ".pdf":
                md = await asyncio.to_thread(
                    ocr_to_markdown_deepseek_pdf,
                    content,
                    ext,
                )
                return PdfToMarkdownResponse(markdown=md)
            if ext in _IMAGE_EXTENSIONS:
                md = await asyncio.to_thread(
                    ocr_to_markdown_deepseek_image,
                    content,
                    ext,
                )
                return PdfToMarkdownResponse(markdown=md)
            raise HTTPException(
                status_code=415,
                detail="Unsupported document type. This endpoint accepts only PDF and image files.",
            )

        # URL path: download to temp and infer from suffix
        if url is not None:
            tmp = await _source_from_url_to_temp(str(url), 120)
            suffix = tmp.suffix.lower()
            if suffix == ".pdf":
                md = await asyncio.to_thread(
                    ocr_to_markdown_deepseek_pdf,
                    tmp,
                    ".pdf",
                )
                return PdfToMarkdownResponse(markdown=md)
            if suffix in _IMAGE_EXTENSIONS:
                md = await asyncio.to_thread(
                    ocr_to_markdown_deepseek_image,
                    tmp,
                    suffix,
                )
                return PdfToMarkdownResponse(markdown=md)
            raise HTTPException(
                status_code=415,
                detail="Unsupported document type. This endpoint accepts only PDF and image files.",
            )

        raise HTTPException(status_code=422, detail="Either file or url must be provided")
    except DocumentFetchError as e:
        raise _http_exception_for_fetch_error(e)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


@router.post("/process-using-glm", response_model=PdfToMarkdownResponse)
async def process_using_glm(
    file: UploadFile | None = File(None),
    url: HttpUrl | None = None,
) -> PdfToMarkdownResponse:
    """
    Convert a PDF or image to markdown using GLM-OCR via Ollama.

    Accepts either an uploaded file or a URL. Only PDF and common image types are supported.
    """
    tmp: Path | None = None
    try:
        # File upload path: use bytes and file_extension
        if file is not None:
            content, ext = await _read_upload(file)
            ext = (ext or "").lower()
            if ext == ".pdf":
                md = await asyncio.to_thread(
                    ocr_to_markdown_glm,
                    content,
                    ".pdf",
                )
                return PdfToMarkdownResponse(markdown=md)
            if ext in _IMAGE_EXTENSIONS:
                md = await asyncio.to_thread(
                    ocr_to_markdown_glm,
                    content,
                    ext,
                )
                return PdfToMarkdownResponse(markdown=md)
            raise HTTPException(
                status_code=415,
                detail="Unsupported document type. This endpoint accepts only PDF and image files.",
            )

        # URL path: download to temp and infer from suffix
        if url is not None:
            tmp = await _source_from_url_to_temp(str(url), 120)
            suffix = tmp.suffix.lower()
            if suffix == ".pdf":
                md = await asyncio.to_thread(
                    ocr_to_markdown_glm,
                    tmp,
                    ".pdf",
                )
                return PdfToMarkdownResponse(markdown=md)
            if suffix in _IMAGE_EXTENSIONS:
                md = await asyncio.to_thread(
                    ocr_to_markdown_glm,
                    tmp,
                    suffix,
                )
                return PdfToMarkdownResponse(markdown=md)
            raise HTTPException(
                status_code=415,
                detail="Unsupported document type. This endpoint accepts only PDF and image files.",
            )

        raise HTTPException(status_code=422, detail="Either file or url must be provided")
    except DocumentFetchError as e:
        raise _http_exception_for_fetch_error(e)
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


@router.post("/chunk", response_model=ChunkResponse)
async def chunk_document(req: ChunkRequest) -> ChunkResponse:
    """Split text into overlapping chunks for RAG."""
    chunks = chunk_text(req.text, req.chunk_size, req.chunk_overlap)
    return ChunkResponse(chunks=chunks)

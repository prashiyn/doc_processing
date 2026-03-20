"""
Document converters.

File-to-markdown (markitdown, docling), iXBRL, PDF/OCR (Docling, GLM-OCR, DeepSeek-OCR),
YouTube transcript, markdown tables to DataFrames.
"""
from __future__ import annotations

import base64
import csv
import json
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from lxml import etree as ET
from markitdown import MarkItDown
from ixbrl_parse.ixbrl import parse

from mistletoe.block_token import Table, TableCell, TableRow
from doc_processing.services.docling_parser import (
    parse_html_using_docling,
    parse_markdown_using_docling,
    parse_pdf_using_docling,
    pdf_to_markdown_docling,
    xbrl_to_markdown,
)

import pandas as pd

from mistletoe import Document
import fitz


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


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


def _default_glm_ocr_config_path() -> Path:
    """Path to default GLM-OCR Ollama config (config/glm_ocr_ollama.yaml)."""
    return _project_root() / "config" / "glm_ocr_ollama.yaml"


def _default_deepseek_ocr_config_path() -> Path:
    """Path to default DeepSeek-OCR Ollama config (config/deepseek_ocr_ollama.yaml)."""
    return _project_root() / "config" / "deepseek_ocr_ollama.yaml"


# Default prompt for DeepSeek-OCR (Ollama). Model expects plain prompt; we ask for Markdown.
# See: https://github.com/deepseek-ai/DeepSeek-OCR (vLLM uses "<image>\\nFree OCR." style).
_DEEPSEEK_OCR_DEFAULT_PROMPT = (
    "Recognize the text in the image and output in Markdown format. "
    "Preserve the original layout (headings, paragraphs, tables, formulas). "
    "Do not fabricate content that does not exist in the image."
)

# Regex to strip DeepSeek-OCR ref/det blocks: <|ref|>...<|/ref|><|det|>...<|/det|>
# See: run_dpsk_ocr_pdf.py / run_dpsk_ocr_image.py (re_match and replacement).
_DEEPSEEK_OCR_REF_DET_PATTERN = re.compile(
    r"<\|ref\|>.*?<\|/ref\|><\|det\|>.*?<\|/det\|>",
    re.DOTALL,
)


def _deepseek_ocr_clean_output(text: str) -> str:
    """
    Post-process DeepSeek-OCR model output: remove ref/det blocks and LaTeX substitutions.

    The vLLM pipeline replaces image refs with markdown image links and strips other refs;
    we strip all ref/det blocks so the result is clean markdown (no embedded bbox metadata).
    See: https://github.com/deepseek-ai/DeepSeek-OCR (run_dpsk_ocr_pdf.py, run_dpsk_ocr_image.py).
    """
    if not text:
        return text
    out = _DEEPSEEK_OCR_REF_DET_PATTERN.sub("", text)
    out = out.replace("\\coloneqq", ":=").replace("\\eqqcolon", "=:")
    out = re.sub(r"\n\n\n+", "\n\n", out)
    return out.strip()


def _load_deepseek_ocr_config(config_path: str | Path | None) -> dict[str, Any]:
    """Load DeepSeek-OCR Ollama config from YAML or return defaults."""
    defaults: dict[str, Any] = {
        "api_host": "localhost",
        "api_port": 11434,
        "api_path": "/api/generate",
        "model": "deepseek-ocr:latest",
        "request_timeout": 300,
    }
    if not config_path:
        return defaults
    path = Path(config_path)
    if not path.exists():
        return defaults
    try:
        import yaml
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        ocr = (data.get("pipeline") or {}).get("ocr_api") or (data.get("ocr_api") or {})
        if ocr.get("api_host") is not None:
            defaults["api_host"] = str(ocr["api_host"])
        if ocr.get("api_port") is not None:
            defaults["api_port"] = int(ocr["api_port"])
        if ocr.get("api_path") is not None:
            defaults["api_path"] = str(ocr["api_path"]).strip() or "/api/generate"
        if ocr.get("model") is not None:
            defaults["model"] = str(ocr["model"])
        if ocr.get("request_timeout") is not None:
            defaults["request_timeout"] = int(ocr["request_timeout"])
    except Exception:
        pass
    return defaults


def _deepseek_ocr_request(
    images_b64: list[str],
    prompt: str,
    *,
    api_host: str = "localhost",
    api_port: int = 11434,
    api_path: str = "/api/generate",
    model: str = "deepseek-ocr:latest",
    request_timeout: int = 300,
) -> str:
    """
    Send one request to Ollama /api/generate for DeepSeek-OCR; return response text.

    Uses the same Ollama generate format as GLM-OCR: model, prompt, stream=False, images.
    """
    import base64
    import requests

    url = f"http://{api_host}:{api_port}{api_path}"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "images": images_b64,
    }
    try:
        r = requests.post(url, json=payload, timeout=request_timeout)
        r.raise_for_status()
        data = r.json()
        text = (data.get("response") or "").strip()
        if not text:
            logging.getLogger(__name__).warning(
                "DeepSeek-OCR Ollama returned 200 but response was empty (model=%s).",
                model,
            )
        return text
    except requests.exceptions.RequestException as e:
        logging.getLogger(__name__).warning(
            "DeepSeek-OCR request failed: %s (url=%s, model=%s)",
            e,
            url,
            model,
        )
        return ""
    except Exception as e:
        logging.getLogger(__name__).warning("DeepSeek-OCR request error: %s", e)
        return ""


def ocr_to_markdown_deepseek_image(
    source: str | Path | bytes,
    file_extension: str | None = None,
    prompt: str | None = None,
    config_path: str | Path | None = None,
    api_host: str | None = None,
    api_port: int | None = None,
    api_path: str | None = None,
    model: str | None = None,
    request_timeout: int | None = None,
    clean_output: bool = True,
) -> str:
    """
    Run DeepSeek-OCR (Ollama) on a single image and return markdown.

    Mirrors the vLLM image flow in run_dpsk_ocr_image.py: one image → one inference
    → optional post-process (strip ref/det blocks). Uses Ollama /api/generate.

    Prerequisites:
    - Ollama with DeepSeek-OCR: ollama pull deepseek-ocr:latest && ollama serve

    Args:
        source: File path (str or Path) to an image, or raw bytes of the image.
        file_extension: Required when source is bytes (e.g. ".png", ".jpg").
        prompt: OCR prompt. If None, uses default Markdown-oriented prompt.
        config_path: Optional path to DeepSeek-OCR config YAML. Ignored if api_* given.
        api_host, api_port, api_path, model, request_timeout: Override config.
        clean_output: If True, strip <|ref|>/<|det|> blocks and normalize (default True).

    Returns:
        Markdown string. Empty string on failure.
    """
    path: Path | None = None
    try:
        if isinstance(source, bytes):
            if not file_extension or not file_extension.strip():
                return ""
            ext = file_extension.strip().lower()
            if not ext.startswith("."):
                ext = f".{ext}"
            path = Path(tempfile.mkdtemp(prefix="deepseek_ocr_")) / f"input{ext}"
            path.write_bytes(source)
        else:
            path = Path(source)

        if not path.exists():
            return ""

        suffix = path.suffix.lower()
        if suffix not in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"):
            return ""

        cfg_path = config_path or _default_deepseek_ocr_config_path()
        cfg = _load_deepseek_ocr_config(cfg_path if Path(cfg_path).exists() else None)
        if api_host is not None:
            cfg["api_host"] = api_host
        if api_port is not None:
            cfg["api_port"] = api_port
        if api_path is not None:
            cfg["api_path"] = api_path
        if model is not None:
            cfg["model"] = model
        if request_timeout is not None:
            cfg["request_timeout"] = request_timeout

        b64 = base64.b64encode(path.read_bytes()).decode("ascii")
        use_prompt = prompt or _DEEPSEEK_OCR_DEFAULT_PROMPT
        text = _deepseek_ocr_request(
            [b64],
            use_prompt,
            api_host=cfg["api_host"],
            api_port=cfg["api_port"],
            api_path=cfg["api_path"],
            model=cfg["model"],
            request_timeout=cfg["request_timeout"],
        )
        if not text:
            return ""
        out = _deepseek_ocr_clean_output(text) if clean_output else text
        return (out or "").strip()
    except Exception as e:
        logging.getLogger(__name__).warning("DeepSeek-OCR image failed: %s", e)
        return ""
    finally:
        if path is not None and isinstance(source, bytes) and path.exists():
            path.unlink(missing_ok=True)
            try:
                shutil.rmtree(path.parent, ignore_errors=True)
            except OSError:
                pass


def ocr_to_markdown_deepseek_pdf(
    source: str | Path | bytes,
    file_extension: str | None = None,
    prompt: str | None = None,
    config_path: str | Path | None = None,
    api_host: str | None = None,
    api_port: int | None = None,
    api_path: str | None = None,
    model: str | None = None,
    request_timeout: int | None = None,
    clean_output: bool = True,
) -> str:
    """
    Run DeepSeek-OCR (Ollama) on a PDF and return markdown (one request per page).

    Mirrors the vLLM PDF flow in run_dpsk_ocr_pdf.py: PDF → images (per page)
    → one inference per image → concatenate and post-process (strip ref/det blocks).

    Prerequisites:
    - Ollama with DeepSeek-OCR: ollama pull deepseek-ocr:latest && ollama serve

    Args:
        source: File path (str or Path) to a PDF, or raw bytes of the PDF.
        file_extension: Required when source is bytes (e.g. ".pdf").
        prompt: OCR prompt. If None, uses default Markdown-oriented prompt.
        config_path: Optional path to DeepSeek-OCR config YAML.
        api_host, api_port, api_path, model, request_timeout: Override config.
        clean_output: If True, strip <|ref|>/<|det|> blocks and normalize (default True).

    Returns:
        Markdown string (pages concatenated with \\n\\n). Empty string on failure.
    """
    path: Path | None = None
    image_paths: list[Path] = []

    try:
        if isinstance(source, bytes):
            if not file_extension or not file_extension.strip():
                return ""
            ext = file_extension.strip().lower()
            if not ext.startswith("."):
                ext = f".{ext}"
            path = Path(tempfile.mkdtemp(prefix="deepseek_ocr_")) / f"input{ext}"
            path.write_bytes(source)
        else:
            path = Path(source)

        if not path.exists():
            return ""

        if path.suffix.lower() != ".pdf":
            return ""

        cfg_path = config_path or _default_deepseek_ocr_config_path()
        cfg = _load_deepseek_ocr_config(cfg_path if (cfg_path and Path(cfg_path).exists()) else None)
        if api_host is not None:
            cfg["api_host"] = api_host
        if api_port is not None:
            cfg["api_port"] = api_port
        if api_path is not None:
            cfg["api_path"] = api_path
        if model is not None:
            cfg["model"] = model
        if request_timeout is not None:
            cfg["request_timeout"] = request_timeout

        temp_dir = Path(tempfile.mkdtemp(prefix="deepseek_ocr_pdf_"))
        image_paths = _pdf_to_image_paths(path, temp_dir)
        if not image_paths:
            return ""

        import base64
        use_prompt = prompt or _DEEPSEEK_OCR_DEFAULT_PROMPT
        parts: list[str] = []
        for i, img_path in enumerate(image_paths):
            b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
            text = _deepseek_ocr_request(
                [b64],
                use_prompt,
                api_host=cfg["api_host"],
                api_port=cfg["api_port"],
                api_path=cfg["api_path"],
                model=cfg["model"],
                request_timeout=cfg["request_timeout"],
            )
            if text:
                cleaned = _deepseek_ocr_clean_output(text) if clean_output else text
                parts.append(cleaned.strip())
            else:
                logging.getLogger(__name__).warning(
                    "DeepSeek-OCR PDF page %s returned empty.", i + 1
                )
        out = "\n\n".join(parts)
        if not out:
            logging.getLogger(__name__).warning(
                "DeepSeek-OCR PDF returned no content. Check Ollama and model deepseek-ocr:latest."
            )
        return out.strip()
    except Exception as e:
        logging.getLogger(__name__).warning("DeepSeek-OCR PDF failed: %s", e)
        return ""
    finally:
        if image_paths:
            temp_dir = image_paths[0].parent
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except OSError:
                pass
        if path is not None and isinstance(source, bytes):
            path.unlink(missing_ok=True)
            try:
                shutil.rmtree(path.parent, ignore_errors=True)
            except OSError:
                pass


def _pdf_to_image_paths(pdf_path: Path, temp_dir: Path) -> list[Path]:
    """Render each PDF page to a PNG in temp_dir; return list of image paths. Requires pymupdf."""

    paths: list[Path] = []
    doc = fitz.open(pdf_path)
    try:
        for i in range(len(doc)):
            page = doc[i]
            pix = page.get_pixmap(dpi=150, alpha=False)
            out = temp_dir / f"page_{i:04d}.png"
            pix.save(str(out))
            paths.append(out)
    finally:
        doc.close()
    return paths


def ocr_to_markdown_glm(
    source: str | Path | bytes,
    file_extension: str | None = None,
    config_path: str | Path | None = None,
) -> str:
    """
    Run GLM-OCR (Ollama) on an image or PDF and return markdown.

    Uses a minimal in-project client that calls Ollama /api/generate (no glmocr package).
    Supports single images and multi-page PDFs (pages rendered to images, one request).
    See: https://github.com/zai-org/GLM-OCR/blob/main/examples/ollama-deploy/README.md

    Prerequisites:
    - Ollama with GLM-OCR: ollama pull glm-ocr:latest && ollama serve
    - Default config: config/glm_ocr_ollama.yaml (api_host=localhost, api_port=11434,
      api_path=/api/generate, model=glm-ocr:latest)

    Args:
        source: File path (str or Path) to an image or PDF, or raw bytes of the file.
        file_extension: Required when source is bytes (e.g. ".pdf", ".png", ".jpg").
        config_path: Path to GLM-OCR config.yaml. If None, uses config/glm_ocr_ollama.yaml.

    Returns:
        Markdown string from the OCR result. Empty string on failure.
    """
    from doc_processing.services.glm_ocr_client import GlmOcrClient

    path: Path | None = None
    image_paths: list[Path] = []

    try:
        if isinstance(source, bytes):
            if not file_extension or not file_extension.strip():
                return ""
            ext = file_extension.strip().lower()
            if not ext.startswith("."):
                ext = f".{ext}"
            path = Path(tempfile.mkdtemp(prefix="glm_ocr_")) / f"input{ext}"
            path.write_bytes(source)
        else:
            path = Path(source)

        if not path.exists():
            return ""

        config_file = Path(config_path) if config_path else _default_glm_ocr_config_path()

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            temp_dir = Path(tempfile.mkdtemp(prefix="glm_ocr_pdf_"))
            image_paths = _pdf_to_image_paths(path, temp_dir)
            if not image_paths:
                return ""
            parse_input: str | list[str] = [str(p) for p in image_paths]
        elif suffix in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"):
            parse_input = str(path)
        else:
            return ""

        with GlmOcrClient(config_path=config_file if config_file.exists() else None) as parser:
            result = parser.parse(parse_input)
            if isinstance(result, list):
                markdown = result[0].markdown_result if result else ""
            else:
                markdown = result.markdown_result
            out = (markdown or "").strip()
            if not out:
                logging.getLogger(__name__).warning(
                    "GLM-OCR returned empty markdown. Check Ollama is running (ollama serve) "
                    "and model is loaded (ollama list); use api_path: /api/generate in config."
                )
            return out
    except Exception as e:
        logging.getLogger(__name__).warning("GLM-OCR failed: %s", e)
        return ""
    finally:
        if image_paths:
            temp_dir = image_paths[0].parent
            try:
                shutil.rmtree(temp_dir, ignore_errors=True)
            except OSError:
                pass
        if path is not None and isinstance(source, bytes):
            path.unlink(missing_ok=True)
            try:
                shutil.rmtree(path.parent, ignore_errors=True)
            except OSError:
                pass

"""GLM OCR parser utilities."""
from __future__ import annotations

import logging
import shutil
import tempfile
from pathlib import Path

import fitz


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_glm_ocr_config_path() -> Path:
    """Path to default GLM-OCR Ollama config (config/glm_ocr_ollama.yaml)."""
    return _project_root() / "config" / "glm_ocr_ollama.yaml"


def _pdf_to_image_paths(pdf_path: Path, temp_dir: Path) -> list[Path]:
    """Render each PDF page to a PNG in temp_dir; return list of image paths."""
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


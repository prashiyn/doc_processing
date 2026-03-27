"""DeepSeek OCR parser utilities."""
from __future__ import annotations

import base64
import logging
import re
import shutil
import tempfile
from pathlib import Path
from typing import Any

import fitz


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _default_deepseek_ocr_config_path() -> Path:
    """Path to default DeepSeek-OCR Ollama config (config/deepseek_ocr_ollama.yaml)."""
    return _project_root() / "config" / "deepseek_ocr_ollama.yaml"


# Default prompt for DeepSeek-OCR (Ollama). Model expects plain prompt; we ask for Markdown.
# See: https://github.com/deepseek-ai/DeepSeek-OCR (vLLM uses "<image>\nFree OCR." style).
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

    Mirrors the vLLM image flow in run_dpsk_ocr_image.py: one image -> one inference
    -> optional post-process (strip ref/det blocks). Uses Ollama /api/generate.
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
    """Run DeepSeek-OCR (Ollama) on a PDF and return markdown (one request per page)."""
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


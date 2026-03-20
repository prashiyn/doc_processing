from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doc_processing.services.docling_parser import pdf_to_markdown_docling  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert a PDF to Markdown using Docling and write output to data/temp/."
    )
    parser.add_argument("pdf_path", help="Path to the input PDF file.")
    parser.add_argument(
        "--backend",
        choices=["easyocr", "vlm"],
        default="easyocr",
        help="Docling backend to use (default: easyocr).",
    )
    parser.add_argument(
        "--merge-tables",
        action="store_true",
        help="Merge split tables and remove header/footer artifacts.",
    )
    parser.add_argument(
        "--no-embed-image",
        action="store_true",
        help="Disable embedding images into Markdown (default embeds images).",
    )
    parser.add_argument(
        "--out-dir",
        default="data/temp",
        help="Output directory for markdown (default: data/temp).",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).expanduser().resolve()
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Input must be an existing .pdf file: {pdf_path}")

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    embed_image = not args.no_embed_image
    md, _ = pdf_to_markdown_docling(
        pdf_path,
        backend=args.backend,
        merge_tables=args.merge_tables,
        embed_image=embed_image,
    )

    out_path = out_dir / f"{pdf_path.stem}.docling.md"
    out_path.write_text(md or "", encoding="utf-8")

    print(f"Wrote markdown to: {out_path}")
    print(f"Markdown length: {len(md or '')} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


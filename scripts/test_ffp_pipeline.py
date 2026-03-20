from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from doc_processing.ffp.pipeline import FinancialFilingsPipeline  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run FinancialFilingsPipeline on a PDF and write chunks JSON."
    )
    parser.add_argument("pdf_path", help="Path to input PDF")
    parser.add_argument(
        "--doc-id",
        default=None,
        help="Document ID used in chunk IDs (default: PDF stem).",
    )
    parser.add_argument(
        "--publish-date",
        default=None,
        help="Optional publish date (YYYY-MM-DD) to attach to chunks.",
    )
    parser.add_argument(
        "--backend",
        choices=["easyocr", "vlm"],
        default="vlm",
        help="PDF parsing backend for Docling (default: vlm).",
    )
    parser.add_argument(
        "--out",
        default="data/temp/chunks.json",
        help="Output JSON path (default: data/temp/chunks.json).",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path).expanduser().resolve()
    if not pdf_path.exists() or pdf_path.suffix.lower() != ".pdf":
        raise SystemExit(f"Input must be an existing PDF: {pdf_path}")

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    doc_id = args.doc_id or pdf_path.stem
    pipeline = FinancialFilingsPipeline()
    chunks = pipeline.run_from_pdf(
        pdf_path,
        doc_id=doc_id,
        publish_date=args.publish_date,
        backend=args.backend,
    )

    out_path.write_text(json.dumps(chunks, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(chunks)} chunks to: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


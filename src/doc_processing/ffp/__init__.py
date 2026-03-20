"""Financial Filings Pipeline (FFP) for structured chunking.

This package implements the chunking pipeline described in docs/DOCUMENT_CHUNKING.md:

- ingestion from Docling conversion results (PDF/Markdown/HTML)
- multimodal conversion of tables and figures to text
- bundle-aware chunk segmentation
- deduplication
- coreference resolution
- section-level metadata summaries

The main entrypoint is `FinancialFilingsPipeline` in `pipeline.py`.
"""


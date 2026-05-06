# doc-processing

Document processing, conversions and chunking for RAG. Uses **docling** (VLM, EasyOCR), **unstructured** (PDF, MD, XLS, PPT, DOC, CSV, images), **ixbrl-parse**, and **markitdown**.

LLM APIs are owned by the standalone `llm-service` and are not exposed from this service.

## Setup

```bash
uv sync
```

## Run

```bash
uv run uvicorn doc_processing.main:app --reload
```

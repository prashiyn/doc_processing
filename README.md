# doc-processing

Document processing, conversions and chunking for RAG. Uses **docling** (VLM, EasyOCR), **unstructured** (PDF, MD, XLS, PPT, DOC, CSV, images), **ixbrl-parse**, **markitdown**, and **litellm** for LLM access.

## Setup

```bash
uv sync
```

## Run

```bash
uv run uvicorn doc_processing.main:app --reload
```

## LLM providers

Configured via env: Groq, Ollama, OpenAI, Anthropic (Claude), Tencent. See `.env.example`.

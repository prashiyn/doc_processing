# Service Split Plan: Document Processing vs LLM Gateway

## Context

The current service combines two distinct capabilities in one runtime:

1. Document processing and ingestion (`/documents/*`)
2. Shared LLM gateway (`/llm/*`) used by other systems

This creates an undesirable flow for downstream systems:

`document -> doc-processing (/documents) -> RAG/agent systems -> doc-processing (/llm) -> response`

As more RAG systems and future agent systems integrate, this coupling causes circular call paths and difficult tracing.

## Goal

Split into two deployables with clear separation of concerns while preserving `/llm` API compatibility.

---

## Confirmed Architectural Decisions (Authoritative)

These decisions are fixed and drive the implementation plan:

1. `/llm` paths and signatures remain exactly the same.
2. Internal auth between services is deferred to a follow-up hardening phase (not part of PR4).
3. If `llm-service` is unavailable, document processing **fails fast**.
4. No in-process LLM fallback mode in local or production environments.
5. Deployment model is two separate deployables.

---

## Target Architecture

## Service A: `llm-service`

Owns:
- `POST /llm/complete`
- `GET /llm/models`
- `POST /llm/embeddings`
- health endpoints

Responsibilities:
- wrapper over LiteLLM provider calls
- reasoning support (`reasoning_effort`)
- structured responses (`response_format`)
- embedding support (including current ollama behavior)
- auth validation (deferred hardening phase)

## Service B: `doc-processing-service`

Owns:
- all `/documents/*`
- health endpoints

Responsibilities:
- parsing, OCR, conversion, chunking, ingestion
- remote calls to `llm-service` over HTTP for all LLM operations
- fail-fast behavior when `llm-service` is unavailable

## Communication Boundary

`doc-processing-service` must not call model providers directly.
It should call only `llm-service` APIs for LLM concerns.

---

## Compatibility Requirements

1. `/llm` request/response contracts must remain unchanged.
2. `/llm` OpenAPI subtree must remain backward compatible.
3. Validation and error semantics for `/llm` endpoints must remain equivalent.
4. Existing external services consuming `/llm` require no integration changes.

---

## File Boundary Guidance

## Move to / owned by `llm-service`
- `src/doc_processing/routers/llm.py`
- `src/doc_processing/llms/client.py`
- `src/doc_processing/llms/embeddings.py`
- `src/doc_processing/llms/config.py`
- `src/doc_processing/llms/groq_ratelimit.py`
- llm config files (owned in `llm-service/src/config/*`)
- llm tests

## Keep in / owned by `doc-processing-service`
- `src/doc_processing/routers/documents.py`
- `src/doc_processing/ffp/**`
- `src/doc_processing/services/**`
- chunking/parser/ocr configs
- document pipeline tests

---

## Runtime Refactor Principle

All current in-process `LLMClient` usage inside doc-processing must be replaced by a single remote runtime client abstraction (e.g., `HttpLLMRuntime`) that calls:

- `POST /llm/complete`
- `POST /llm/embeddings`

No in-process fallback implementation should be introduced.

---

## Migration Phases

## Phase 0: Contract Freeze

- Snapshot `/llm` OpenAPI contract
- Add contract checks to prevent breaking changes

## Phase 1: Standalone LLM Service

- Build and run independent `llm-service`
- Keep `/llm` contract unchanged
- Verify llm tests pass in standalone runtime

## Phase 2: Remote LLM Runtime + Pipeline Switch

- Add remote runtime client in doc-processing
- Switch all doc-processing LLM operations to remote `/llm`
- Validate fail-fast behavior on llm unavailability

## Phase 3: Final Separation Cleanup

- Remove `/llm` mounting from doc-processing app
- remove direct provider dependencies from doc-processing where possible
- finalize service-specific configs, CI, and deploy artifacts

---

## Testing and Acceptance

## Contract checks
- `/llm` OpenAPI compatibility checks
- `/llm` request/response schema regression tests

## Behavior checks
- reasoning/structured output/embeddings behavior parity
- auth success/failure tests (deferred hardening phase)
- fail-fast outage behavior tests

## Pipeline checks
- document output regression suite on representative corpus
- ensure no in-process LLM call paths remain

---

## Final Topology

- `llm-service` (dedicated deployable)
- `doc-processing-service` (dedicated deployable)

This removes circular service flow and makes tracing, ownership, and future extensibility (including agent workloads) significantly cleaner.

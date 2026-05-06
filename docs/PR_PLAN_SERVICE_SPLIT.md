# PR Plan: Service Split Execution

This PR plan is fully aligned to `docs/SERVICE_SPLIT_PLAN.md` and its fixed decisions.

## Fixed Decisions (Must Not Drift)

1. `/llm` API paths/signatures remain unchanged.
2. Internal service-to-service auth is deferred and tracked separately from PR4.
3. On `llm-service` outage, doc pipeline fails fast.
4. No in-process LLM fallback mode anywhere.
5. Two independent deployables.

---

## Phase to PR Mapping

- **Phase 0** -> PR1
- **Phase 1** -> PR2
- **Phase 2** -> PR3 + PR4
- **Phase 3** -> PR5

---

## Current Status

- [x] PR1 complete
- [x] PR2 complete
- [x] PR3 complete
- [x] PR4 complete (pipeline switched to remote `/llm`, no auth changes)
- [x] PR5 complete

---

## Ownership Split

## Team A (LLM Service)
- standalone `llm-service`
- `/llm` API compatibility
- auth enforcement on `/llm`
- llm contract tests

## Team B (Doc Processing)
- document pipeline migration to remote `/llm`
- fail-fast behavior
- removal of direct LLM provider coupling in doc paths

## Joint
- CI wiring
- OpenAPI contract gates
- deployment docs and runbooks

---

## PR1 (Phase 0): Contract Freeze

### Goal
Lock `/llm` contract before structural changes.

### Scope
- Capture `/llm` OpenAPI baseline snapshot
- Add compatibility test/diff gate for `/llm` subtree
- Define allowed non-breaking schema drift (e.g., ordering/description text)

### Acceptance Criteria
- CI fails on breaking `/llm` contract changes
- baseline snapshot committed and reproducible

### Owner
- Team A (primary), Team B (review)

---

## PR2 (Phase 1): Standalone `llm-service`

### Goal
Run `/llm` endpoints as an independent deployable with unchanged behavior.

### Scope
- Create standalone app entrypoint for `llm-service`
- Keep endpoints exactly same:
  - `POST /llm/complete`
  - `GET /llm/models`
  - `POST /llm/embeddings`
- Ensure tests execute against standalone app

### Acceptance Criteria
- `/llm` contract remains compatible with PR1 baseline
- `llm-service` runs without `/documents` runtime
- llm test suite passes

### Owner
- Team A

---

## PR3 (Phase 2): Remote Runtime Client in Doc Service

### Goal
Introduce a strict remote runtime client for doc -> llm communication.

### Scope
- Add remote client (e.g., `HttpLLMRuntime`) mapping to:
  - `POST /llm/complete`
  - `POST /llm/embeddings`
- Add config for base URL and timeout
- Add request/response mapping + error normalization
- Add tracing/correlation header propagation

### Acceptance Criteria
- Client unit tests pass (success, timeout, 4xx/5xx, malformed payload)
- no in-process fallback path added

### Owner
- Team B

---

## PR4 (Phase 2): Pipeline Switch to Remote `/llm` (No Auth)

### Goal
Move all document pipeline LLM calls to remote `/llm` without changing API contracts.

### Scope
- Refactor all FFP/doc LLM stages to use remote runtime client with use-case mapping
- Remove direct in-process `LLMClient` usage from doc-processing code paths
- Enforce fail-fast behavior when `llm-service` is unavailable

### Acceptance Criteria
- all doc LLM stages use remote runtime
- llm outage causes deterministic fail-fast behavior

### Owner
- Team B

---

## PR5 (Phase 3): Final Separation Cleanup

### Goal
Complete service separation and harden deployables.

### Scope
- Remove `/llm` router mounting from doc-processing app
- Split service-specific config/dependency ownership
- Add boundary checks to prevent cross-domain imports
- Update runbooks/deployment docs for two-service model

### Acceptance Criteria
- two deployables build and run independently
- boundary checks pass
- docs and operational playbooks updated

### Owner
- Team A + Team B

---

## Cross-PR Quality Gates

For every PR:
- tests for changed area
- lint/type checks
- OpenAPI generation/checks

Additional gates:
- PR1+: `/llm` contract gate mandatory
- PR4+: fail-fast outage tests mandatory
- PR5+: import-boundary checks mandatory

---

## Agent-Friendly Execution Template

Use this prompt per PR:

"Implement PR<NUM> from `docs/PR_PLAN_SERVICE_SPLIT.md`.
Constraints:
1. Preserve `/llm` contract exactly.
2. Do not add or keep in-process LLM fallback mode.
3. Add tests for acceptance criteria of PR<NUM>.
4. Regenerate/update OpenAPI artifacts if needed.
5. Return a checklist-mapped diff summary."

---

## Rollout Validation Checklist

- [x] `/llm` contract unchanged
- [x] `llm-service` standalone healthy
- [x] doc-processing calls only remote `/llm`
- [ ] internal auth plan finalized for follow-up phase
- [x] llm outage fail-fast verified
- [x] traces show clean non-circular service flow

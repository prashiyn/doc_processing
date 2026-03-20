# LLM usage (LiteLLM)

This package provides multi-provider LLM access for NSE corporate data parsing and analysis. It uses [LiteLLM](https://docs.litellm.ai/) with a thin wrapper; API keys are set via **environment variables** (e.g. in `.env` or `config_dir/api_keys.env`). Config files (`llms.yaml`, `groq_limits.yaml`) live in the project config directory (see `doc_processing.config.get_config_dir()`; override with `DOC_PROCESSING_CONFIG_DIR`).

---

## Quick start

```python
from llms import LLMClient

client = LLMClient()
out = client.complete([{"role": "user", "content": "Summarize in one line: Bonus 1:1, ex-date Jan 2025."}])
print(out)
```

Ensure at least one provider’s key is set (see below) and, if needed, set `default_model` in config_dir `llms.yaml` to a model you have access to.

---

## Provider setup

Add the relevant variables to `config/api_keys.env` (or `.env`). Then add the model(s) to `config/llms.yaml` under `models` and/or set `default_model` / `fallback_model`.

### OpenAI

| Env var | Example model in `config/llms.yaml` |
|--------|--------------------------------------|
| `OPENAI_API_KEY` | `openai/gpt-4o-mini`, `openai/gpt-4o`, `openai/gpt-3.5-turbo` |

```bash
# config/api_keys.env
OPENAI_API_KEY=sk-...
```

### Anthropic (Claude)

| Env var | Example model in `config/llms.yaml` |
|--------|--------------------------------------|
| `ANTHROPIC_API_KEY` | `anthropic/claude-3-5-sonnet-20241022`, `anthropic/claude-3-haiku-20240307` |

```bash
# config/api_keys.env
ANTHROPIC_API_KEY=sk-ant-...
```

### Groq

Fast inference; no key in repo, set in env.

| Env var | Example model in `config/llms.yaml` |
|--------|--------------------------------------|
| `GROQ_API_KEY` | `groq/llama-3.3-70b-versatile`, `groq/llama-3.1-8b-instant` |

```bash
# config/api_keys.env
GROQ_API_KEY=gsk_...
```

In `config/llms.yaml` you can set e.g. `default_model: "groq/llama-3.1-8b-instant"`.

**Groq rate limiting:** All requests that use a model starting with `groq/` are automatically rate-limited (RPM and RPD) so loops stay within [Groq limits](https://console.groq.com/settings/limits). Limits are read from `config/groq_limits.yaml` (see [Groq rate limits docs](https://console.groq.com/docs/rate-limits)). The limiter blocks until a request is allowed before calling the API.

### Local Ollama

No API key. Ensure [Ollama](https://ollama.ai/) is running locally (default: `http://localhost:11434`).

| Env var (optional) | Example model in `config/llms.yaml` |
|-------------------|--------------------------------------|
| — | `ollama/llama3.2`, `ollama/llama3.1`, `ollama/llama2`, `ollama/mistral` |

```bash
# Optional: custom base (default is http://localhost:11434)
# OLLAMA_API_BASE=http://localhost:11434
```

In `config/llms.yaml`:

```yaml
default_model: "ollama/llama3.2"
models:
  - "ollama/llama3.2"
  - "ollama/llama3.1"
```

### MiniMax (e.g. MiniMax-M2.5)

| Env var | Example model in `config/llms.yaml` |
|--------|--------------------------------------|
| `MINIMAX_API_KEY` | `minimax/MiniMax-M2.5`, `minimax/MiniMax-M2.5-lightning`, `minimax/MiniMax-M2.1` |
| `MINIMAX_API_BASE` (optional) | `https://api.minimax.io/v1` (OpenAI-compatible endpoint) |

```bash
# config/api_keys.env
MINIMAX_API_KEY=your-minimax-api-key
# MINIMAX_API_BASE=https://api.minimax.io/v1
```

In `config/llms.yaml`:

```yaml
default_model: "minimax/MiniMax-M2.5"
# or faster: minimax/MiniMax-M2.5-lightning
models:
  - "minimax/MiniMax-M2.5"
  - "minimax/MiniMax-M2.5-lightning"
  - "minimax/MiniMax-M2.1"
```

---

## Config: `config/llms.yaml`

- **`default_model`** – Used by `LLMClient` when no `model` is passed.
- **`fallback_model`** – Used by `complete_with_fallback` / `acomplete_with_fallback` if the primary call fails.
- **`models`** – List of model strings you may use (for reference or tooling).
- **`analysis`** – Optional per-task models for NSE analysis:
  - `summarization_model`
  - `classification_model`
  - `extraction_model`

Example with multiple providers:

```yaml
default_model: "groq/llama-3.1-8b-instant"
fallback_model: "openai/gpt-3.5-turbo"

models:
  - "openai/gpt-4o-mini"
  - "openai/gpt-3.5-turbo"
  - "anthropic/claude-3-5-sonnet-20241022"
  - "groq/llama-3.3-70b-versatile"
  - "groq/llama-3.1-8b-instant"
  - "ollama/llama3.2"
  - "minimax/MiniMax-M2.5"

analysis:
  summarization_model: "groq/llama-3.1-8b-instant"
  classification_model: "openai/gpt-4o-mini"
  extraction_model: "minimax/MiniMax-M2.5"
```

---

## Usage

### Direct client

```python
from llms import LLMClient

client = LLMClient()

# Sync
text = client.complete([{"role": "user", "content": "Hello"}])
text = client.complete_with_fallback([{"role": "user", "content": "Hello"}])

# With a specific model
text = client.complete(
    [{"role": "user", "content": "Hello"}],
    model="anthropic/claude-3-haiku-20240307",
)

# Async
import asyncio
async def run():
    text = await client.acomplete([{"role": "user", "content": "Hello"}])
    return text
asyncio.run(run())
```

### Agent runner (multi-step)

```python
from llms.agents import AgentRunner

runner = AgentRunner()
steps = [
    {"prompt": "List the corporate action types in: {{context}}", "role": "extraction"},
    {"prompt": "Summarize in one paragraph: {{context}}", "role": "summarization"},
]
outputs = runner.run_steps(steps, initial_context="Bonus 1:1, Dividend Rs 5, Split 1:2")
# outputs[0] = extraction, outputs[1] = summary
```

### NSE corporate analysis

NSE analysis lives in `data.interim.nse_analysis` (uses this package for LLM calls). Pass rows from QuestDB / MCP into the analysis helpers:

```python
from data.interim.nse_analysis import (
    analyze_corporate_actions,
    parse_corporate_action_subjects,
    summarize_announcements,
    classify_board_meeting_purposes,
)

# Rows from corporate_actions table
ca_rows = [
    {"underlying_symbol": "RELIANCE", "subject": "Bonus 1:1", "ex_date": "2025-01-01", ...},
]
result = analyze_corporate_actions(ca_rows)
# result["summary"], result["action_types"], result["by_symbol"]

# Board meetings: uses tickers/nse_event_map.csv + Ollama for canonical_events, period_ending, result_type, canonical_subevent, tags
bm_rows = [{"description": "...", "purpose": "...", "underlying_symbol": "BEL", "br_date": ...}]
classified = classify_board_meeting_purposes(bm_rows)
```

See `data/interim/nse_analysis.py` and `data/utils/utils.py` (event map filtering), and `docs/llms_setup.md`.

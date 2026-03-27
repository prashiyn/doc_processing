"""
LLM access for parsing and analyzing NSE corporate data.

Uses LiteLLM for multi-provider support. Configure models in config_dir/llms.yaml
and API keys in .env or config_dir/api_keys.env (see docs/llms_setup.md).
"""
from dotenv import load_dotenv

# Load .env from cwd first so DOC_PROCESSING_CONFIG_DIR can be set
load_dotenv()
# Then overlay config_dir/api_keys.env if present
from doc_processing.config import get_config_dir
_config_dir = get_config_dir()
_api_keys_env = _config_dir / "api_keys.env"
if _api_keys_env.exists():
    load_dotenv(_api_keys_env)

from doc_processing.llms.client import LLMClient
from doc_processing.llms.embeddings import EmbeddingClient
from doc_processing.llms.config import get_llm_config

__all__ = ["LLMClient", "EmbeddingClient", "get_llm_config"]

# NSE analysis lives in data.interim: from data.interim.nse_analysis import ...
# Agents: from llms.agents import AgentRunner

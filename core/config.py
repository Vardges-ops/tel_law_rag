import os
import configparser
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

_cfg = configparser.ConfigParser(interpolation=None)
_cfg.read(Path(__file__).parent.parent / "config.cfg")

# --- secrets (env only, never in config.cfg) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
CEREBRAS_API_KEY = os.environ.get("CEREBRAS_API_KEY", "")

# --- non-secret knobs ---
QDRANT_URL = os.environ.get("QDRANT_URL") or None  # None => embedded mode
EMBEDDING_MODEL = _cfg.get("MODELS", "embedding_model", fallback="BAAI/bge-m3")
RERANKER_MODEL = _cfg.get("MODELS", "reranker_model", fallback="BAAI/bge-reranker-v2-m3")

TOP_K_RETRIEVE = _cfg.getint("RETRIEVAL", "top_k_retrieve", fallback=10)
TOP_K_RERANK = _cfg.getint("RETRIEVAL", "top_k_rerank", fallback=2)
RRF_K = _cfg.getint("RETRIEVAL", "rrf_k", fallback=60)

GENERATION_TEMPERATURE = _cfg.getfloat("GENERATION", "temperature", fallback=0.0)

LOG_LEVEL = _cfg.get("LOGGING", "log_level", fallback="INFO")

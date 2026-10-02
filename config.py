"""
Configuration constants for the MOJ Laws scraper.
"""

import os

# Base URLs
BASE_URL = "https://laws.moj.gov.sa"
PORTAL_URL = f"{BASE_URL}/ar/legislations-regulations"
LEGISLATION_WEB_URL = f"{BASE_URL}/ar/legislation"

# Gateway API Endpoints (Official public API used by the MOJ Nuxt portal)
GATEWAY_BASE_URL = "https://laws-gateway.moj.gov.sa/apis/legislations/v1"
SEARCH_API_URL = f"{GATEWAY_BASE_URL}/statute/section-search"
DETAIL_API_URL = f"{GATEWAY_BASE_URL}/statute/get-Statute-gateway-Detail"

# Search defaults
DEFAULT_PAGE_SIZE = 9
DEFAULT_SORTING = 7  # 7: Newest first, 1: Alphabetical, 8: Most visited
DEFAULT_MAX_PAGES = 50

# Rate limiting & timeouts
REQUEST_DELAY_SECONDS = 1.5  # Polite delay between requests
PAGE_LOAD_TIMEOUT_MS = 30000  # 30 seconds for initial portal load

# Output paths (absolute, so the scripts work from any working directory)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "data")
JSON_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "legislation.json")
CSV_OUTPUT_FILE = os.path.join(OUTPUT_DIR, "legislation.csv")
DATABASE_PATH = os.path.join(OUTPUT_DIR, "saudi_laws.db")

# Retry policy for gateway API calls (search pages and legislation details)
MAX_RETRIES = 3
RETRY_DELAYS_SECONDS = [3, 6, 12]

# Browser settings
HEADLESS = True
VIEWPORT_WIDTH = 1280
VIEWPORT_HEIGHT = 900

# Logging
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
LOG_LEVEL = "INFO"

# ---------------------------------------------------------------------------
# RAG settings
# ---------------------------------------------------------------------------

# Local vector store (numpy embeddings + JSON metadata)
INDEX_DIR = os.path.join(OUTPUT_DIR, "index")

# Chunking (characters). Articles are kept whole when they fit; long articles
# are split on paragraph boundaries; very short neighbouring articles are merged.
CHUNK_MAX_CHARS = 1500
CHUNK_MIN_CHARS = 300

# On Render only the project directory is kept from the build step to runtime, so
# the embedding model cache (downloaded during the build) must live inside it.
# Locally (RENDER not set) the normal Hugging Face cache in the user profile is used.
if os.environ.get("RENDER") and not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = os.path.join(BASE_DIR, ".hf_cache")

# Multilingual embedding model (Arabic supported), runs locally on CPU.
# e5 models expect "query: " / "passage: " prefixes (handled in rag/vector_store.py).
EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# Retrieval
TOP_K = 5
# Minimum cosine similarity for a chunk to count as relevant. Below this the
# system answers that the available legal data is insufficient (without calling
# the LLM). Calibrated for multilingual-e5-small on 28 Arabic questions:
# in-scope questions scored 0.850-0.893 (one casual phrasing 0.832), out-of-scope
# ones (VAT, marriage, inheritance, traffic, ...) 0.802-0.838.
MIN_RELEVANCE_SCORE = float(os.environ.get("MIN_RELEVANCE_SCORE", "0.845"))

# LLM (Anthropic Claude). Optional: without ANTHROPIC_API_KEY the system returns
# the retrieved passages with citations instead of a generated answer.
LLM_MODEL = os.environ.get("LLM_MODEL", "claude-opus-5-5")
LLM_MAX_TOKENS = 16000
LLM_TIMEOUT_SECONDS = 120
AGENT_MAX_STEPS = 6

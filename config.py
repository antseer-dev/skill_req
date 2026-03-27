import os
from dotenv import load_dotenv

load_dotenv()

# ── API Keys ─────────────────────────────────────────────
LUNARCRUSH_KEY = os.environ.get("LUNARCRUSH_KEY", "")
OPENROUTER_KEY = os.environ.get("OPENROUTER_KEY", "")
# ── Base URLs ─────────────────────────────────────────────
LUNARCRUSH_BASE = "https://lunarcrush.com"
OPENROUTER_BASE = "https://openrouter.ai/api/v1"

# ── Models ────────────────────────────────────────────────
MODEL_EMBED  = "qwen/qwen3-embedding-8b"
MODEL_BULK   = "google/gemini-2.5-flash"
MODEL_REPORT = "anthropic/claude-sonnet-4-5"   # via OpenRouter

# ── LunarCrush ────────────────────────────────────────────
DEFAULT_TOPICS = [
    "bitcoin", "ethereum", "solana", "defi",
    "nft", "crypto", "web3", "altcoin",
]
MAX_POSTS_PER_TOPIC = 50
LUNARCRUSH_CONCURRENCY = 5
CACHE_TTL_HOURS = 24

# ── Clustering ────────────────────────────────────────────
UMAP_COMPONENTS     = 10
UMAP_N_NEIGHBORS    = 15
HDBSCAN_MIN_CLUSTER = 5
HDBSCAN_MIN_SAMPLES = 3

UMAP_NEED_COMPONENTS     = 8
HDBSCAN_NEED_MIN_CLUSTER = 4
HDBSCAN_NEED_MIN_SAMPLES = 2

# ── Pipeline ──────────────────────────────────────────────
GEMINI_CONCURRENCY = 20
EMBED_BATCH_SIZE   = 50
TOP_N_SKILLS       = 10

# ── Storage ───────────────────────────────────────────────
DB_PATH     = "storage/cache.db"
OUTPUT_DIR  = "output"

# ── Callback ──────────────────────────────────────────────
CALLBACK_URL = os.environ.get(
    "CALLBACK_URL",
    "http://localhost:8575/api/v1/workflow/task/callback",
)

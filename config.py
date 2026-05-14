"""集中配置项。所有可调参数都在这里。"""
from pathlib import Path

# ===== 阿里云百炼 =====
DASHSCOPE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_MODEL = "qwen-plus"
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIM = 1024
EMBEDDING_BATCH = 10
EMBEDDING_MAX_TOKENS = 8192

# ===== LLM 调用控制 =====
LLM_TIMEOUT_SEC = 60
LLM_RETRY_MAX = 3
LLM_JSON_RETRY = 1
LLM_BACKOFF_BASE = 2.0

# ===== 检索 =====
TOP_K = 3

# ===== 路径 =====
ROOT = Path(__file__).resolve().parent
VECTOR_DB_DIR = ROOT / "vector_db"
LOG_DIR = ROOT / "logs"
PROGRESS_DIR = ROOT / "progress"
FAILED_DIR = ROOT / "failed"
OUTPUT_DIR = ROOT / "output"

CHROMA_COLLECTION = "chunks"

# ===== 业务 =====
CATEGORIES = [
    "案例", "產品", "投保規則", "健康核保",
    "財務核保", "繳費", "行政規則", "一般查詢",
]
TYPE_VALUES = ["text", "img", "table"]

# ===== 健壮性 =====
FAIL_THRESHOLD = 5
SUPPORTING_FACTS_OVERLAP_MIN = 0.5

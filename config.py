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

# ===== v3 当前活跃配置 =====
INPUT_FILE = "inputs/chunks_v3.jsonl"
CHROMA_COLLECTION = "chunks_v3"
INTERMEDIATE_QA_JSONL = "output/chunks_v3.qa.jsonl"
OUTPUT_XLSX = "output/qa_test_full_v3.xlsx"

# ===== 历史保留（切换 dataset 时把上面几行替换为下面对应的即可）=====
# v1/v2 (旧 chunks)：
# INPUT_FILE = "inputs/chunks.jsonl"
# CHROMA_COLLECTION = "chunks"
# INTERMEDIATE_QA_JSONL = "output/chunks.qa.jsonl"
# OUTPUT_XLSX = "output/qa_test_full_v2.xlsx"

# ===== 业务 =====
CATEGORIES = [
    "案例", "產品", "投保規則", "健康核保",
    "財務核保", "繳費", "行政規則", "一般查詢",
]
TYPE_VALUES = ["text", "img", "table"]

# ===== 健壮性 =====
FAIL_THRESHOLD = 5
SUPPORTING_FACTS_OVERLAP_MIN = 0.5

# ===== 批量模式（10 文件）=====
BATCH_INPUT_DIR = "inputs/batch"
BATCH_OUTPUT_DIR = "output/per_file"
BATCH_AGGREGATED_XLSX = "output/qa_test_all.xlsx"
BATCH_MANIFEST = "progress/_batch.manifest.json"
BATCH_FAILED_LOG = "failed/_batch.failed.json"
BATCH_FAIL_FILE_THRESHOLD = 3      # 累计失败文件 ≥ 3 暂停
CHROMA_COLLECTION_BATCH = "chunks_batch"

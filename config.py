"""
全局配置文件。所有可调参数都集中在这里，避免代码里散落 magic number。
"""
# ⚠️ Windows-only workaround:必须在 import pandas / sentence_transformers
# 之前先加载 pyarrow,否则 sentence_transformers → datasets → pandas.compat.pyarrow
# 链路触发 pyarrow C 扩展的 access violation(Windows 11,Python 3.10,
# pyarrow 24.0.0 + pandas 2.3.3 的 ABI 冲突,顺序敏感)。
# 不要删这行。config.py 是所有入口的第一个 import,放在这里能确保所有 entry 都生效。
# 如果未来 pyarrow / pandas 升级后此冲突消失,本行可以移除,但请先在 Windows 上
# 跑 `python -c "import sentence_transformers; from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu').encode(['hi'])"` 验证。
import pyarrow  # noqa: F401 — preload before pandas; see comment above

from pathlib import Path
import os
from dotenv import load_dotenv

# ============================================================
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent

# 从项目根的 .env 加载环境变量(HF_TOKEN / 各家 LLM API key)
# .env 是本地的,git-ignored,绝对不要 commit
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus_static"
POISON_DIR = DATA_DIR / "poison_sets"
CACHE_DIR = DATA_DIR / "cache"
QUERY_FILE = DATA_DIR / "test_queries.yaml"

# 静态库分两层（ADJ-001）：
#   - BASE: 和 query 主题对齐的真实文档（任务 A 的产出,Brisbane）
#   - BACKGROUND: 通用噪声,模拟真实世界 corpus 规模（MS MARCO 抽样）
BASE_CORPUS_FILE = CORPUS_DIR / "brisbane_corpus.json"
BACKGROUND_CORPUS_FILE = CORPUS_DIR / "msmarco_background.json"

# 缓存的 FAISS 索引文件名（combined = BASE + BACKGROUND 合并后的索引）
FAISS_CACHE = CACHE_DIR / "static_index_combined.faiss"
EMBED_CACHE = CACHE_DIR / "static_embeddings_combined.npy"
DOCS_CACHE = CACHE_DIR / "static_documents_combined.json"

# ============================================================
# 模型配置
# ============================================================
# Embedding 模型（本地，免费）
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # all-MiniLM-L6-v2 的维度

# 设备：cuda / cpu / mps
# 如果有 GPU 就用 GPU，否则 CPU
EMBEDDING_DEVICE = "cuda"  # 改成 "cpu" 如果你没 GPU

# ============================================================
# Retrieval 配置
# ============================================================
TOP_K_1 = 10   # dense retriever 返回多少个 (k_1)
TOP_K_2 = 5    # LLM reranker 返回多少个 (k_2)

# ============================================================
# LLM 配置（全部通过 OpenRouter 路由）
# ============================================================
# 旧设计是 4 家直接接各自 SDK(anthropic / openai / google-generativeai / requests),
# 现在统一走 OpenRouter:1 个 key, 1 笔账单, 1 个 client class。
# Model 命名遵循 OpenRouter 的 "provider/model" 格式。
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

AVAILABLE_LLMS = {
    "claude": {
        "provider": "openrouter",
        "model": "anthropic/claude-sonnet-4.5",
        "enabled": True,
    },
    "gpt4o": {
        "provider": "openrouter",
        "model": "openai/gpt-4o-mini",
        "enabled": True,
    },
    "gemini": {
        "provider": "openrouter",
        # 2026-05-13 切换:v1 主实验用的 google/gemini-2.0-flash-001 在 OpenRouter
        # listing 上标注 "Going away 2026-06-01",capacity 收缩导致 vertex endpoint
        # uptime 73%、ai-studio endpoint 撞 RPM hard cap(retry-with-backoff 也救不回)。
        # 切到 gemini-2.5-flash-lite:同价位($0.10 in / $0.40 out per 1M),
        # 3 个 endpoint 全 healthy(uptime ≥99.78%),版本反而升级一代。
        # Report Methodology 需 disclose model 切换。
        "model": "google/gemini-2.5-flash-lite",
        "enabled": True,
        # 保留 provider hint 作为保险(2.5-flash-lite 当前 3 endpoint 都健康,
        # 但 pin 到 ai-studio 避免未来 vertex 抽风影响复现)。
        "openrouter_provider": {
            "order": ["google-ai-studio"],
            "allow_fallbacks": False,
        },
    },
    "llama": {
        "provider": "openrouter",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "enabled": True,
    },
}

# Generator 用的固定模型（避免笛卡尔积）
GENERATOR_LLM = "claude"

# ============================================================
# Poison 生成(ADJ-002)
# ============================================================
# 用于 ADJ-002 5 种 attack 的 LLM(spec §1.5)。锁定到 gpt-4o,
# 不要用 gpt-chat-latest(会自动更新,破坏实验复现性)。
POISON_GENERATOR_MODEL = "openai/gpt-4o"

# query_targets.yaml: 每条 query 的 poison_target + target_type(spec §1.1)
QUERY_TARGETS_FILE = DATA_DIR / "query_targets.yaml"

# keyword_stuffing 用的 variants 缓存(spec §2.5)
KEYWORD_VARIANTS_CACHE = CACHE_DIR / "keyword_variants.json"

# ============================================================
# UI 配置
# ============================================================
APP_TITLE = "RAG Poisoning Visualizer"
SIDEBAR_PAGES = ["Dashboard", "Attack Module", "Experiment", "History"]

"""
Global configuration. All tunable parameters live here so the code stays free of magic numbers.
全局配置文件。所有可调参数集中在这里,避免代码里散落 magic number。
"""
# Windows-only workaround: pyarrow must load BEFORE pandas / sentence_transformers,
# otherwise the sentence_transformers → datasets → pandas.compat.pyarrow chain
# triggers a pyarrow C-extension access violation (Windows 11 + Python 3.10 +
# pyarrow 24.0.0 + pandas 2.3.3 ABI clash; order-sensitive).
# Do NOT remove this line. config.py is the first import on every entry point,
# so placing it here guarantees the preload fires for all entries.
# If a future pyarrow / pandas upgrade resolves this, the line can be removed,
# but first verify on Windows with:
#     python -c "import sentence_transformers; from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu').encode(['hi'])"
#
# Windows-only 解决方案:必须在 import pandas / sentence_transformers 之前先加载
# pyarrow,否则会触发 pyarrow C 扩展的 access violation(Win11 + Py3.10 +
# pyarrow 24.0.0 + pandas 2.3.3 的 ABI 冲突,顺序敏感)。不要删这行。config.py
# 是所有入口的第一个 import,放在这里能确保所有 entry 都生效。未来如果 pyarrow /
# pandas 升级后冲突消失可移除,先在 Windows 上跑上述命令验证。
import pyarrow  # noqa: F401 — preload before pandas; see comment above

from pathlib import Path
import os
from dotenv import load_dotenv

# ============================================================
# Paths
# 路径配置
# ============================================================
PROJECT_ROOT = Path(__file__).parent

# Load environment variables (HF_TOKEN / per-vendor LLM keys) from .env at the project root.
# .env is local and git-ignored — NEVER commit it.
# 从项目根的 .env 加载环境变量(HF_TOKEN / 各家 LLM key);.env 本地 git-ignored,绝对不要 commit。
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus_static"
POISON_DIR = DATA_DIR / "poison_sets"
CACHE_DIR = DATA_DIR / "cache"
QUERY_FILE = DATA_DIR / "test_queries.yaml"

# Static corpus is two-tiered (ADJ-001):
#   - BASE:       topic-aligned real documents (Brisbane).
#   - BACKGROUND: generic noise to simulate real-world corpus scale (MS MARCO sample).
# 静态库分两层(ADJ-001):BASE = Brisbane 主题对齐真实文档;BACKGROUND = MS MARCO 通用噪声。
BASE_CORPUS_FILE = CORPUS_DIR / "brisbane_corpus.json"
BACKGROUND_CORPUS_FILE = CORPUS_DIR / "msmarco_background.json"

# Cached FAISS index file names (combined = BASE + BACKGROUND merged).
# 缓存的 FAISS 索引文件名(combined = BASE + BACKGROUND 合并后的索引)。
FAISS_CACHE = CACHE_DIR / "static_index_combined.faiss"
EMBED_CACHE = CACHE_DIR / "static_embeddings_combined.npy"
DOCS_CACHE = CACHE_DIR / "static_documents_combined.json"

# ============================================================
# Models
# 模型配置
# ============================================================
# Local, free embedding model.
# 本地免费的 embedding 模型。
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384  # dim of all-MiniLM-L6-v2

# Device: cuda / cpu / mps. Set to "cpu" if no GPU.
# 设备:cuda / cpu / mps。没 GPU 改 "cpu"。
EMBEDDING_DEVICE = "cuda"

# ============================================================
# Retrieval
# 检索参数
# ============================================================
TOP_K_1 = 10   # dense retriever output size (k_1)
TOP_K_2 = 5    # LLM reranker output size (k_2)

# ============================================================
# LLM configuration (all routed through OpenRouter)
# LLM 配置(全部走 OpenRouter)
# ============================================================
# The original design used per-vendor SDKs (anthropic / openai / google-generativeai / requests);
# the unified OpenRouter path collapses that into one key, one bill, one client class.
# Model names follow OpenRouter's "provider/model" convention.
# 旧设计是 4 家直接接各自 SDK,现在统一走 OpenRouter:1 个 key / 1 笔账单 / 1 个 client class。
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
        # Model switch (2026-05-13): the v1 main run used google/gemini-2.0-flash-001,
        # which OpenRouter listed as "Going away 2026-06-01". Capacity contraction made
        # the vertex endpoint uptime ~73% and the ai-studio endpoint hit RPM hard caps
        # (retry-with-backoff couldn't save it). Switched to gemini-2.5-flash-lite:
        # same price tier ($0.10 in / $0.40 out per 1M), 3 endpoints all healthy
        # (uptime >= 99.78%), and a generation newer. Report Methodology must disclose
        # the swap.
        # 2026-05-13 切换:v1 主实验用的 gemini-2.0-flash-001 在 OpenRouter 上标注
        # "Going away 2026-06-01",capacity 收缩导致 vertex uptime 73% + ai-studio
        # 撞 RPM hard cap。切到 gemini-2.5-flash-lite:同价位,3 endpoint 全 healthy,
        # 版本反而升级一代。Report Methodology 需 disclose model 切换。
        "model": "google/gemini-2.5-flash-lite",
        "enabled": True,
        # Provider hint kept as insurance: 2.5-flash-lite's 3 endpoints are all healthy
        # now, but pinning to ai-studio guards against future vertex flakiness affecting
        # reproducibility.
        # 保留 provider hint 作为保险:2.5-flash-lite 当前 3 endpoint 都健康,但 pin 到
        # ai-studio 避免未来 vertex 抽风影响复现。
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

# Fixed generator model (avoids reranker × generator cartesian product).
# Generator 用的固定模型(避免笛卡尔积)。
GENERATOR_LLM = "claude"

# ============================================================
# Poison generation (ADJ-002)
# Poison 生成
# ============================================================
# LLM used by ADJ-002's 5 attacks. Pinned to gpt-4o — DO NOT use gpt-chat-latest
# (auto-updating breaks experiment reproducibility).
# ADJ-002 的 5 种 attack 用的 LLM。锁定到 gpt-4o,不要用 gpt-chat-latest(会自动
# 更新,破坏实验复现性)。
POISON_GENERATOR_MODEL = "openai/gpt-4o"

# query_targets.yaml: per-query poison_target + target_type.
# query_targets.yaml:每条 query 的 poison_target + target_type。
QUERY_TARGETS_FILE = DATA_DIR / "query_targets.yaml"

# Variants cache for keyword_stuffing.
# keyword_stuffing 用的 variants 缓存。
KEYWORD_VARIANTS_CACHE = CACHE_DIR / "keyword_variants.json"

# ============================================================
# UI
# UI 配置
# ============================================================
APP_TITLE = "RAG Poisoning Visualizer"
SIDEBAR_PAGES = ["Dashboard", "Attack Module", "Experiment", "History"]

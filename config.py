"""
全局配置文件。所有可调参数都集中在这里，避免代码里散落 magic number。
"""
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

# 默认语料库文件名（任务 A 的产出）
DEFAULT_CORPUS_FILE = CORPUS_DIR / "brisbane_corpus.json"

# 缓存的 FAISS 索引文件名
FAISS_CACHE = CACHE_DIR / "static_index.faiss"
EMBED_CACHE = CACHE_DIR / "static_embeddings.npy"
DOCS_CACHE = CACHE_DIR / "static_documents.json"

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
# LLM 配置（明天再实际使用）
# ============================================================
# 从环境变量读 API key（避免提交到 git）
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")

# 可用的 LLM 列表（reranker 用）
AVAILABLE_LLMS = {
    "claude": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-20250514",
        "enabled": False,  # 明天接 API 时改 True
    },
    "gpt4o": {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "enabled": False,
    },
    "gemini": {
        "provider": "google",
        "model": "gemini-2.0-flash-exp",
        "enabled": False,
    },
    "llama": {
        "provider": "openrouter",
        "model": "meta-llama/llama-3.3-70b-instruct",
        "enabled": False,
    },
}

# Generator 用的固定模型（避免笛卡尔积）
GENERATOR_LLM = "claude"

# ============================================================
# UI 配置
# ============================================================
APP_TITLE = "RAG Poisoning Visualizer"
SIDEBAR_PAGES = ["Dashboard", "Attack Module", "Experiment", "History"]

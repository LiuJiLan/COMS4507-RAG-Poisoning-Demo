"""
快速 CLI 实验脚本 —— 不走 GUI，直接看 ranking 结果。

开发期主要在这里跑：
- 改下面 QUERY / POISON_SET 变量
- PyCharm 绿三角运行
- 终端立刻看到 clean vs poisoned 的两阶段排名 + 攻击指标

GUI 只在 demo 那天用。
"""
import os
import sys
import warnings
from pathlib import Path

# === 噪音抑制（必须在 HF / transformers / sentence-transformers 加载前）===
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

# Windows 控制台默认 GBK,无法 encode ☣ 等 Unicode 标记。强制 UTF-8,encoding fail 走替换。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# 让 scripts 能 import 项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
# WARNING 级就够,不要 INFO(避免 sentence-transformers 大量 HEAD 日志刷屏)
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

# 抑制 transformers 内部 logger 的噪音(yolos / zoedepth 等 lazy-load warning)
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# 抑制 huggingface_hub 的 HF_TOKEN unauthenticated 警告(走自己的 logger,不是 Python warnings)
from huggingface_hub.utils import logging as hub_logging
hub_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import config
from src.corpus import load_corpus, load_poison_set
from src.pipeline import RAGPipeline
from src.evaluator import format_ranking_table


# ======================================================================
# 改这里跑不同实验
# ======================================================================
QUERY = "best Chinese restaurant in Brisbane"
POISON_SET = "P_demo"      # 对应 data/poison_sets/<这里>.json
INCLUDE_GENERATOR = False  # stub 阶段开了也只是占位输出
# ======================================================================


def _section(title: str) -> None:
    print()
    print(f"--- {title} ---")
    print()


def main() -> None:
    # ---- 构建 / 加载 pipeline ----
    pipeline = RAGPipeline(
        embedding_model=config.EMBEDDING_MODEL,
        embedding_device=config.EMBEDDING_DEVICE,
        top_k_1=config.TOP_K_1,
        top_k_2=config.TOP_K_2,
    )

    if config.FAISS_CACHE.exists() and config.DOCS_CACHE.exists():
        pipeline.load_cached_index(config.FAISS_CACHE, config.DOCS_CACHE)
    else:
        docs = load_corpus(config.DEFAULT_CORPUS_FILE)
        pipeline.initialize(docs)

    # ---- 加载 poison set ----
    poison_path = config.POISON_DIR / f"{POISON_SET}.json"
    if not poison_path.exists():
        raise FileNotFoundError(
            f"Poison set not found: {poison_path}\n"
            f"Available: {sorted(p.stem for p in config.POISON_DIR.glob('*.json'))}"
        )
    poison_docs = load_poison_set(poison_path)

    # ---- Header ----
    print()
    print("=" * 72)
    print(f"Query:        {QUERY}")
    print(f"Poison set:   {POISON_SET}  ({len(poison_docs)} doc{'s' if len(poison_docs) != 1 else ''})")
    print(f"Clean corpus: {pipeline.retriever.n_clean} docs")
    print("=" * 72)

    # ---- 跑实验 ----
    result = pipeline.run_experiment(
        query=QUERY,
        poison_docs=poison_docs,
        include_generator=INCLUDE_GENERATOR,
    )

    # ---- Stage 1: dense retrieval ----
    _section(f"Stage 1: dense retrieval (top-{config.TOP_K_1})")
    print("[Clean]")
    print(format_ranking_table(result.top_k1_clean))
    print()
    print("[Poisoned]")
    print(format_ranking_table(result.top_k1_poisoned))
    print()
    m1 = result.metrics_k1
    print(f"k1 metrics: {m1}")
    if m1.score_gap is not None:
        print(f"  score gap (best_poison - clean_top1): {m1.score_gap:+.4f}")
    if m1.displaced_docs:
        print(f"  displaced docs: {m1.displaced_docs}")

    # ---- Stage 2: LLM reranker ----
    _section(f"Stage 2: LLM reranker (top-{config.TOP_K_2})")
    print("[Clean]")
    print(format_ranking_table(result.top_k2_clean))
    print()
    print("[Poisoned]")
    print(format_ranking_table(result.top_k2_poisoned))
    print()
    m2 = result.metrics_k2
    print(f"k2 metrics: {m2}")
    if m2.score_gap is not None:
        print(f"  score gap (best_poison - clean_top1): {m2.score_gap:+.4f}")
    if m2.displaced_docs:
        print(f"  displaced docs: {m2.displaced_docs}")

    # ---- Stage 3: generator (optional) ----
    if INCLUDE_GENERATOR:
        _section("Stage 3: generated answers")
        print("[Clean answer]")
        print(result.answer_clean)
        print()
        print("[Poisoned answer]")
        print(result.answer_poisoned)

    print()


if __name__ == "__main__":
    main()

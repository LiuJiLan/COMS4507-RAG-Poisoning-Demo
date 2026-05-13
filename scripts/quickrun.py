"""
Quick CLI experiment script — bypasses the GUI and prints rankings directly.
快速 CLI 实验脚本,不走 GUI,直接看 ranking 结果。

Primary dev-time workflow:
- Edit the QUERY / POISON_SET / RERANKER_LLM constants below.
- Hit "run" in PyCharm.
- See the clean vs poisoned two-stage rankings + attack metrics in the terminal.

The GUI is reserved for the live demo.
开发期主要用这个跑;改下面的 QUERY / POISON_SET / RERANKER_LLM 即可。GUI 只在 demo 那天用。
"""
import os
import sys
import warnings
from pathlib import Path

# === Noise suppression — must precede HF / transformers / sentence-transformers imports ===
# === 噪音抑制(必须在 HF / transformers / sentence-transformers 加载前) ===
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")

# Windows console defaults to GBK and cannot encode ☣ etc.; force UTF-8 with replace.
# Windows 控制台默认 GBK,无法 encode ☣ 等 Unicode 标记;强制 UTF-8,encoding fail 走替换。
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

# Allow `python scripts/quickrun.py` to import top-level project modules.
# 让 `python scripts/quickrun.py` 能 import 项目模块。
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
# WARNING level is enough — INFO floods stdout with sentence-transformers HEAD logs.
# WARNING 级就够,不要 INFO(避免 sentence-transformers HEAD 日志刷屏)。
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

# Silence the noisy transformers lazy-load warnings (yolos / zoedepth etc.).
# 抑制 transformers 内部 logger 的噪音(yolos / zoedepth 等 lazy-load warning)。
from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()

# Silence huggingface_hub's HF_TOKEN unauthenticated warning (its own logger, not Python warnings).
# 抑制 huggingface_hub 的 HF_TOKEN unauthenticated 警告(走自己的 logger,不是 Python warnings)。
from huggingface_hub.utils import logging as hub_logging
hub_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import config
from src.corpus import load_corpus, load_poison_set
from src.pipeline import RAGPipeline
from src.evaluator import format_ranking_table
from src.llm_clients import make_client


# ======================================================================
# Edit here to run different experiments.
# 改这里跑不同实验。
# ======================================================================
QUERY = "best Chinese restaurant in Brisbane"
POISON_SET = "P_demo"        # matches data/poison_sets/<this>.json
RERANKER_LLM = "claude"      # config.AVAILABLE_LLMS key: claude / gpt4o / gemini / llama
USE_STUB_RERANKER = False    # True skips the real LLM; reranker uses stub (no API quota)
INCLUDE_GENERATOR = False    # True runs the generator (extra LLM call for the answer)
# ======================================================================


def _section(title: str) -> None:
    print()
    print(f"--- {title} ---")
    print()


def main() -> None:
    # ---- Build LLM clients ----
    reranker_client = make_client(
        config.AVAILABLE_LLMS[RERANKER_LLM],
        use_stub=USE_STUB_RERANKER,
    )
    # Use stub for the generator when not invoked, to save quota.
    # generator 不调用时用 stub,省 quota。
    generator_client = make_client(
        config.AVAILABLE_LLMS[config.GENERATOR_LLM],
        use_stub=not INCLUDE_GENERATOR,
    )

    # ---- Build / load pipeline ----
    pipeline = RAGPipeline(
        embedding_model=config.EMBEDDING_MODEL,
        embedding_device=config.EMBEDDING_DEVICE,
        top_k_1=config.TOP_K_1,
        top_k_2=config.TOP_K_2,
        reranker_client=reranker_client,
        generator_client=generator_client,
    )

    if config.FAISS_CACHE.exists() and config.DOCS_CACHE.exists():
        pipeline.load_cached_index(config.FAISS_CACHE, config.DOCS_CACHE)
    else:
        base_docs = load_corpus(config.BASE_CORPUS_FILE)
        background_docs = load_corpus(config.BACKGROUND_CORPUS_FILE)
        pipeline.initialize(base_docs + background_docs)

    # ---- Load poison set ----
    poison_path = config.POISON_DIR / f"{POISON_SET}.json"
    if not poison_path.exists():
        raise FileNotFoundError(
            f"Poison set not found: {poison_path}\n"
            f"Available: {sorted(p.stem for p in config.POISON_DIR.glob('*.json'))}"
        )
    poison_docs = load_poison_set(poison_path)

    # ---- Header ----
    gen_suffix = "" if INCLUDE_GENERATOR else " (skipped)"
    print()
    print("=" * 72)
    print(f"Query:        {QUERY}")
    print(f"Poison set:   {POISON_SET}  ({len(poison_docs)} doc{'s' if len(poison_docs) != 1 else ''})")
    print(f"Clean corpus: {pipeline.retriever.n_clean} docs")
    print(f"Reranker:     {reranker_client!r}")
    print(f"Generator:    {generator_client!r}{gen_suffix}")
    print("=" * 72)

    # ---- Run the experiment ----
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

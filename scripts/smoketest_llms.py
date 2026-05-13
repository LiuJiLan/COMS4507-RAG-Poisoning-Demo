"""
Smoke test for the OpenRouter LLM integration.
OpenRouter LLM 集成的冒烟测试。

Verifies that every enabled LLM in config.AVAILABLE_LLMS can be called via
OpenRouter and that its response parses cleanly through the reranker.
检查 config.AVAILABLE_LLMS 里所有 enabled 的 LLM 都能通过 OpenRouter 调通,
且响应能被 reranker 解析。

Not a research experiment — just an engineering regression check. Useful when:
- OpenRouter renames a model (e.g. a deprecated ID is replaced) — run this and
  the failing ID surfaces immediately.
- Adding a new LLM to AVAILABLE_LLMS — smoke first before broadening experiments.

只是工程回归测试。OpenRouter 改 model 命名 / 加新 LLM 时跑一遍,立刻定位哪个 ID 失效。

For each LLM it runs 1 clean + 1 poisoned rerank (2 API calls), verifying:
1. The API call succeeds (no 404 / 401 / 5xx / timeout).
2. The response parses via LLMReranker._parse_ranking.
3. The rerank actually reorders (i.e. the LLM was reached, not a silent fallback).

每个 LLM 跑 1 次 clean + 1 次 poisoned rerank(共 2 次 API call),验证:API 成功 /
响应可解析 / 确实改了顺序(说明真调到了 LLM,不是 silent fallback)。

Expected cost (4 models × 2 calls): ~$0.02 USD; runtime 30-60s.
"""
import os
import sys
import warnings
import time
from pathlib import Path

# === Noise suppression — must precede HF / transformers / sentence-transformers imports ===
# === 噪音抑制(必须在 HF / transformers / sentence-transformers 加载前) ===
os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
warnings.filterwarnings("ignore", message=".*HF_TOKEN.*")
warnings.filterwarnings("ignore", message=".*unauthenticated requests.*")
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
from huggingface_hub.utils import logging as hub_logging
hub_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

import config
from src.corpus import load_poison_set
from src.pipeline import RAGPipeline
from src.llm_clients import make_client


# ======================================================================
# Fixed query + poison set; only the LLM dimension varies.
# 固定 query + poison,只比较 LLM 维度。
# ======================================================================
QUERY = "best Chinese restaurant in Brisbane"
POISON_SET = "P_demo"
# ======================================================================


def main() -> None:
    # ---- Build pipeline once; swap reranker.llm per iteration. ----
    # ---- 构建 pipeline 一次,后面只替换 reranker.llm。 ----
    pipeline = RAGPipeline(
        embedding_model=config.EMBEDDING_MODEL,
        embedding_device=config.EMBEDDING_DEVICE,
        top_k_1=config.TOP_K_1,
        top_k_2=config.TOP_K_2,
    )
    pipeline.load_cached_index(config.FAISS_CACHE, config.DOCS_CACHE)
    poison_docs = load_poison_set(config.POISON_DIR / f"{POISON_SET}.json")

    enabled = [(k, v) for k, v in config.AVAILABLE_LLMS.items() if v.get("enabled")]

    print(f"Smoke testing {len(enabled)} LLM(s)")
    print(f"  query:        {QUERY}")
    print(f"  poison set:   {POISON_SET}  ({len(poison_docs)} docs)")
    print(f"  clean corpus: {pipeline.retriever.n_clean} docs")
    print("=" * 76)

    results = []
    for key, cfg in enabled:
        model = cfg["model"]
        print(f"\n[{key}] {model}")

        try:
            pipeline.reranker.llm = make_client(cfg)
            t0 = time.time()
            result = pipeline.run_experiment(
                query=QUERY,
                poison_docs=poison_docs,
                include_generator=False,
            )
            elapsed = time.time() - t0

            # Check that rerank actually reordered (otherwise may be a silent fallback).
            # 检查 rerank 确实改了顺序(否则可能是 silent fallback 到原序)。
            k1_top_ids = [r.doc.doc_id for r in result.top_k1_clean[:config.TOP_K_2]]
            k2_ids = [r.doc.doc_id for r in result.top_k2_clean]
            reordered = k1_top_ids != k2_ids

            top3_p = result.top_k2_poisoned[:3]
            top3_repr = ", ".join(
                ("☣ " if r.doc.is_poison else "  ") + r.doc.title[:28]
                for r in top3_p
            )

            print(f"  status:      OK ({elapsed:.1f}s for 2 API calls)")
            print(f"  reordered:   {reordered}  (clean side, k1 top-{config.TOP_K_2} vs k2)")
            print(f"  k2 poison top-3: {top3_repr}")
            print(f"  k2 metrics:  {result.metrics_k2}")

            results.append((key, "OK", reordered, elapsed, result.metrics_k2))

        except Exception as e:
            print(f"  status:      FAIL  ({type(e).__name__}: {e})")
            results.append((key, "FAIL", False, 0.0, None))

    # ---- Summary table ----
    print("\n" + "=" * 76)
    print(f"{'LLM':<8} {'Status':<7} {'Reordered':<11} {'Time':<8} {'Attack k2'}")
    print("-" * 76)
    for key, status, reordered, elapsed, m in results:
        if m is None:
            attack = "—"
        else:
            attack = f"{'SUCCESS' if m.poison_in_topk else 'fail'}"
            if m.poison_rank is not None:
                attack += f" (rank={m.poison_rank})"
        print(f"{key:<8} {status:<7} {str(reordered):<11} {elapsed:>5.1f}s  {attack}")
    print()


if __name__ == "__main__":
    main()

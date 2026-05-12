"""
scripts/run_experiment.py — 批量实验脚本

跑 (query × poison_set × reranker_LLM) 笛卡尔积,每个组合调用
pipeline.run_experiment(),把指标 row by row 写到 CSV。

用法:
  python scripts/run_experiment.py                   # 全跑(用真 LLM)
  python scripts/run_experiment.py --stub            # 全跑用 stub(不消耗 quota,验证脚本逻辑)
  python scripts/run_experiment.py --limit 5         # 只跑前 5 个组合
  python scripts/run_experiment.py --llms claude llama   # 只跑这两家
  python scripts/run_experiment.py --output data/results/my.csv

预算估算(完整真实跑):
  ~30 query × ~5 poison × 4 LLM = 600 组合,每组合 2 个 API call → ~1200 调用
  ~$2-4 USD,~1 小时(主要等 claude latency)

设计要点:
- pipeline 只构建一次,后续每次只 swap reranker.llm
- 每行 CSV 含 query/poison/llm 标识 + k1+k2 各自的 metrics + 耗时
- error 列允许部分失败不中断整批
"""
import argparse
import csv
import os
import sys
import time
import warnings
from datetime import datetime
from itertools import product
from pathlib import Path

# === 噪音抑制(必须在 HF / transformers / sentence-transformers 加载前)===
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

import yaml
import config
from src.corpus import load_poison_set
from src.pipeline import RAGPipeline
from src.llm_clients import make_client


CSV_FIELDS = [
    "timestamp", "query_id", "query", "poison_set", "n_poison_docs",
    "reranker_llm", "reranker_model",
    "k1_attack_success", "k1_poison_rank", "k1_n_poison", "k1_displaced", "k1_score_gap",
    "k2_attack_success", "k2_poison_rank", "k2_n_poison", "k2_displaced", "k2_score_gap",
    "elapsed_sec", "error",
]


def load_queries(path: Path) -> list[dict]:
    """从 YAML 加载 query 列表。期望 list of dicts,每个 dict 含 query_id + query。"""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, list):
        raise ValueError(f"{path} 顶层应是 list,实际是 {type(data).__name__}")
    for i, item in enumerate(data):
        if not isinstance(item, dict) or "query" not in item:
            raise ValueError(f"{path} 第 {i} 条缺少 query 字段")
    return data


def load_all_poison_sets(poison_dir: Path) -> dict:
    """加载 data/poison_sets/ 下所有 .json,返回 {name: [Document,...]}。"""
    out = {}
    for p in sorted(poison_dir.glob("*.json")):
        try:
            out[p.stem] = load_poison_set(p)
        except Exception as e:
            print(f"  WARN: skip {p.name} ({e})")
    return out


def run_one(pipeline: RAGPipeline,
            query_item: dict,
            poison_set_name: str,
            poison_docs: list,
            llm_key: str,
            llm_cfg: dict,
            use_stub: bool) -> dict:
    """跑一个 (query, poison_set, llm) 组合,返回一行 dict。"""
    pipeline.reranker.llm = make_client(llm_cfg, use_stub=use_stub)

    row = {f: "" for f in CSV_FIELDS}
    row.update({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query_id": query_item.get("query_id", ""),
        "query": query_item["query"],
        "poison_set": poison_set_name,
        "n_poison_docs": len(poison_docs),
        "reranker_llm": llm_key,
        "reranker_model": "stub" if use_stub else llm_cfg["model"],
    })

    t0 = time.time()
    try:
        result = pipeline.run_experiment(
            query=query_item["query"],
            poison_docs=poison_docs,
            include_generator=False,
        )
        elapsed = time.time() - t0

        m1, m2 = result.metrics_k1, result.metrics_k2
        row.update({
            "k1_attack_success": m1.poison_in_topk,
            "k1_poison_rank": m1.poison_rank if m1.poison_rank is not None else "",
            "k1_n_poison": m1.n_poison_in_topk,
            "k1_displaced": len(m1.displaced_docs),
            "k1_score_gap": f"{m1.score_gap:.4f}" if m1.score_gap is not None else "",
            "k2_attack_success": m2.poison_in_topk,
            "k2_poison_rank": m2.poison_rank if m2.poison_rank is not None else "",
            "k2_n_poison": m2.n_poison_in_topk,
            "k2_displaced": len(m2.displaced_docs),
            "k2_score_gap": f"{m2.score_gap:.4f}" if m2.score_gap is not None else "",
            "elapsed_sec": f"{elapsed:.2f}",
        })
    except Exception as e:
        row["error"] = f"{type(e).__name__}: {e}"
        row["elapsed_sec"] = f"{time.time() - t0:.2f}"

    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="批量 RAG 投毒实验")
    parser.add_argument("--stub", action="store_true",
                        help="用 StubLLMClient(不消耗 quota,只测脚本逻辑)")
    parser.add_argument("--limit", type=int, default=None,
                        help="只跑前 N 个组合(冒烟 / 调试用)")
    parser.add_argument("--llms", nargs="+", default=None,
                        help='只跑指定 LLM keys,例:--llms claude llama。默认 = config 里所有 enabled')
    parser.add_argument("--output", type=Path, default=None,
                        help="输出 CSV 路径(默认 data/results/expr_<timestamp>.csv)")
    args = parser.parse_args()

    # ---- 加载输入 ----
    queries = load_queries(config.QUERY_FILE)
    poison_sets = load_all_poison_sets(config.POISON_DIR)
    if not poison_sets:
        raise SystemExit(f"未找到任何 poison set in {config.POISON_DIR}")

    all_enabled = [k for k, v in config.AVAILABLE_LLMS.items() if v.get("enabled")]
    if args.llms:
        invalid = set(args.llms) - set(config.AVAILABLE_LLMS.keys())
        if invalid:
            raise SystemExit(
                f"未知 LLM keys: {invalid}。Available: {list(config.AVAILABLE_LLMS.keys())}"
            )
        llm_keys = args.llms
    else:
        llm_keys = all_enabled

    # ---- Output path ----
    if args.output:
        out_path = args.output
    else:
        out_dir = config.DATA_DIR / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = out_dir / f"expr_{stamp}.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- 笛卡尔积 ----
    combos = list(product(queries, poison_sets.items(), llm_keys))
    if args.limit:
        combos = combos[: args.limit]

    total = len(combos)
    avg_cost = 0.0 if args.stub else 0.003   # 4 模型平均
    est_cost = total * 2 * avg_cost

    print("Experiment plan:")
    print(f"  queries:       {len(queries)}")
    print(f"  poison sets:   {len(poison_sets)} ({list(poison_sets.keys())})")
    print(f"  LLMs:          {len(llm_keys)} ({llm_keys})")
    print(f"  combinations:  {total}{'  (LIMITED)' if args.limit else ''}")
    print(f"  mode:          {'STUB (no API)' if args.stub else 'REAL LLM'}")
    print(f"  est. cost:     ~${est_cost:.2f} USD")
    print(f"  output:        {out_path}")
    print()

    # ---- Pipeline ----
    pipeline = RAGPipeline(
        embedding_model=config.EMBEDDING_MODEL,
        embedding_device=config.EMBEDDING_DEVICE,
        top_k_1=config.TOP_K_1,
        top_k_2=config.TOP_K_2,
    )
    pipeline.load_cached_index(config.FAISS_CACHE, config.DOCS_CACHE)

    # ---- 跑 ----
    rows = []
    t_start = time.time()
    for i, (query_item, (ps_name, ps_docs), llm_key) in enumerate(combos, start=1):
        llm_cfg = config.AVAILABLE_LLMS[llm_key]
        qid = query_item.get("query_id", "?")
        print(f"[{i:>3}/{total}] q={qid:<6} poison={ps_name:<10} llm={llm_key:<8}",
              end="", flush=True)
        row = run_one(pipeline, query_item, ps_name, ps_docs, llm_key, llm_cfg, args.stub)
        rows.append(row)
        if row["error"]:
            marker = f"ERROR ({row['error'][:40]})"
        else:
            marker = f"k2={row['k2_attack_success']}"
        print(f"  → {marker:<25} ({row['elapsed_sec']}s)")
    total_elapsed = time.time() - t_start

    # ---- CSV ----
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    # ---- 汇总 ----
    n_error = sum(1 for r in rows if r["error"])
    ok_rows = [r for r in rows if not r["error"]]
    n_k1 = sum(1 for r in ok_rows if r["k1_attack_success"] is True)
    n_k2 = sum(1 for r in ok_rows if r["k2_attack_success"] is True)

    print()
    print("=" * 70)
    print(f"Done in {total_elapsed:.1f}s")
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"  k1 attack success: {n_k1} / {len(ok_rows)}")
    print(f"  k2 attack success: {n_k2} / {len(ok_rows)}")
    if n_error:
        print(f"  errors: {n_error}  (see 'error' column)")


if __name__ == "__main__":
    main()

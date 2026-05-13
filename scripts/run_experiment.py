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
import json
import os
import subprocess
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
logger = logging.getLogger("run_experiment")

from transformers.utils import logging as hf_logging
hf_logging.set_verbosity_error()
from huggingface_hub.utils import logging as hub_logging
hub_logging.set_verbosity_error()
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)


def fmt_dur(seconds: float) -> str:
    """Format duration as MM:SS or H:MM:SS."""
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def attach_errors_log(out_path: Path) -> Path:
    """Add a WARNING+ FileHandler to the root logger, writing to expr_<ts>.errors.log.

    Captures everything that goes wrong (LLM under-output, API errors, parse failures,
    run_one exceptions) so the run is reviewable without scrolling stdout.
    """
    err_path = out_path.with_suffix(".errors.log")
    h = logging.FileHandler(err_path, encoding="utf-8", mode="w")
    h.setLevel(logging.WARNING)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s"))
    logging.getLogger().addHandler(h)
    return err_path

import yaml
import config
from src.corpus import load_poison_set
from src.pipeline import RAGPipeline
from src.llm_clients import make_client


CSV_FIELDS = [
    "timestamp", "query_id", "query", "poison_set", "n_poison_docs",
    "reranker_llm", "reranker_model",
    "k1_attack_success", "k1_poison_rank", "k1_n_poison", "k1_displaced", "k1_displaced_ids", "k1_score_gap",
    "k2_attack_success", "k2_poison_rank", "k2_n_poison", "k2_displaced", "k2_displaced_ids", "k2_score_gap",
    "reranker_padded_clean", "reranker_padded_poisoned",
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


def load_failed_combos(csv_path: Path, queries: list, poison_sets: dict, all_llms: set,
                       padded_threshold: int | None = None) -> list:
    """读上次 CSV 重建待 retry 的 (query_item, (ps_name, ps_docs), llm_key) tuple 列表。

    匹配规则:
      - `error` 列非空 → always retry(API call 抛了异常)
      - 若 `padded_threshold` 给定且 CSV 有 `reranker_padded_clean/_poisoned` 列,
        则 max(两列) >= threshold 的 row 也 retry(LLM under-output 或完全 fallback)
    """
    qid_to_item = {q.get("query_id"): q for q in queries}
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        combos = []
        seen_keys: set = set()
        n_by_error = 0
        n_by_padded = 0
        for r in reader:
            qualifies = False
            if r.get("error"):
                qualifies = True
                n_by_error += 1
            elif padded_threshold is not None:
                pc = _as_int(r.get("reranker_padded_clean"))
                pp = _as_int(r.get("reranker_padded_poisoned"))
                if max(pc, pp) >= padded_threshold:
                    qualifies = True
                    n_by_padded += 1
            if not qualifies:
                continue
            qid, ps_name, llm_key = r["query_id"], r["poison_set"], r["reranker_llm"]
            key = (qid, ps_name, llm_key)
            if key in seen_keys:
                continue  # dedupe(同一 row 在 CSV 里应该只出现 1 次,防御性)
            seen_keys.add(key)
            q_item = qid_to_item.get(qid)
            if q_item is None or ps_name not in poison_sets or llm_key not in all_llms:
                print(f"  WARN: skip row (qid={qid}, ps={ps_name}, llm={llm_key}) — not resolvable")
                continue
            combos.append((q_item, (ps_name, poison_sets[ps_name]), llm_key))
    print(f"[retry] qualifying rows: {n_by_error} by error + {n_by_padded} by padded(threshold={padded_threshold}) = {len(combos)} combos")
    return combos


def _as_int(s: str | None) -> int:
    if s is None or s == "":
        return 0
    try:
        return int(s)
    except (ValueError, TypeError):
        return 0


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent.parent, text=True
        ).strip()
    except Exception:
        return "unknown"


def write_meta_sidecar(csv_path: Path, args, llm_keys, poison_sets, queries,
                       n_combos: int, n_errors: int, total_elapsed: float) -> Path:
    """跟 CSV 同名的 .meta.json sidecar,存复现需要的全部上下文。"""
    meta_path = csv_path.with_suffix(".meta.json")
    meta = {
        "csv": csv_path.name,
        "errors_log": csv_path.with_suffix(".errors.log").name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_hash": git_hash(),
        "cmd": " ".join(sys.argv),
        "args": vars(args) | {"output": str(args.output) if args.output else None,
                              "retry_errors": str(args.retry_errors) if args.retry_errors else None},
        "config_snapshot": {
            "EMBEDDING_MODEL": config.EMBEDDING_MODEL,
            "EMBEDDING_DEVICE": config.EMBEDDING_DEVICE,
            "TOP_K_1": config.TOP_K_1,
            "TOP_K_2": config.TOP_K_2,
            "QUERY_FILE": str(config.QUERY_FILE),
            "POISON_DIR": str(config.POISON_DIR),
            "FAISS_CACHE": str(config.FAISS_CACHE),
            "rerankers": {
                k: {"model": config.AVAILABLE_LLMS[k].get("model"),
                    "provider": config.AVAILABLE_LLMS[k].get("provider")}
                for k in llm_keys
            },
        },
        "inputs": {
            "n_queries": len(queries),
            "poison_sets": {k: len(v) for k, v in poison_sets.items()},
            "llm_keys": llm_keys,
        },
        "run": {
            "n_combos": n_combos,
            "n_errors": n_errors,
            "elapsed_sec": round(total_elapsed, 2),
        },
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)
    return meta_path


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
            "k1_displaced_ids": "|".join(m1.displaced_docs),
            "k1_score_gap": f"{m1.score_gap:.4f}" if m1.score_gap is not None else "",
            "k2_attack_success": m2.poison_in_topk,
            "k2_poison_rank": m2.poison_rank if m2.poison_rank is not None else "",
            "k2_n_poison": m2.n_poison_in_topk,
            "k2_displaced": len(m2.displaced_docs),
            "k2_displaced_ids": "|".join(m2.displaced_docs),
            "k2_score_gap": f"{m2.score_gap:.4f}" if m2.score_gap is not None else "",
            "reranker_padded_clean": result.reranker_padded_clean,
            "reranker_padded_poisoned": result.reranker_padded_poisoned,
            "elapsed_sec": f"{elapsed:.2f}",
        })
    except Exception as e:
        logger.exception(
            "run_one failed: query_id=%s poison=%s llm=%s",
            query_item.get("query_id"), poison_set_name, llm_key,
        )
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
    parser.add_argument("--retry-errors", type=Path, default=None, dest="retry_errors",
                        help="读上次 CSV,只重跑 error 行(忽略 --limit / 笛卡尔积)")
    parser.add_argument("--padded-threshold", type=int, default=None, dest="padded_threshold",
                        help="配合 --retry-errors:max(reranker_padded_clean, _poisoned) >= N "
                             "的 row 也重跑(N=1 重跑所有 anomaly;N=10 只重跑完全 fallback)")
    args = parser.parse_args()
    if args.padded_threshold is not None and args.retry_errors is None:
        parser.error("--padded-threshold requires --retry-errors")

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

    # ---- Errors log sidecar ----
    errors_log = attach_errors_log(out_path)

    # ---- 笛卡尔积 / retry-errors ----
    if args.retry_errors:
        combos = load_failed_combos(args.retry_errors, queries, poison_sets, set(llm_keys),
                                    padded_threshold=args.padded_threshold)
        if not combos:
            raise SystemExit(f"No resolvable rows in {args.retry_errors}")
    else:
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
    print(f"  errors log:    {errors_log.name}")
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
        elapsed_total = time.time() - t_start
        eta = (elapsed_total / i) * (total - i)
        print(f"  → {marker:<25} ({row['elapsed_sec']}s)  [t={fmt_dur(elapsed_total)} eta={fmt_dur(eta)}]")
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

    # ---- Sidecar meta ----
    meta_path = write_meta_sidecar(
        out_path, args, llm_keys, poison_sets, queries,
        n_combos=len(rows), n_errors=n_error, total_elapsed=total_elapsed,
    )

    print()
    print("=" * 70)
    print(f"Done in {total_elapsed:.1f}s")
    print(f"Wrote {len(rows)} rows to {out_path}")
    print(f"Wrote meta sidecar to {meta_path.name}")
    print(f"Wrote errors log to {errors_log.name} (size {errors_log.stat().st_size} bytes)")
    print(f"  k1 attack success: {n_k1} / {len(ok_rows)}")
    print(f"  k2 attack success: {n_k2} / {len(ok_rows)}")
    if n_error:
        print(f"  errors: {n_error}  (see 'error' column; re-run with --retry-errors {out_path.name})")


if __name__ == "__main__":
    main()

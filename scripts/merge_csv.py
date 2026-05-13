"""
scripts/merge_csv.py — 融合多个 expr_*.csv,按 (query_id, poison_set, reranker_llm)
triplet 匹配;后到的 row 覆盖前到的(命令行右边的 CSV 优先级高)。

用法:
    python scripts/merge_csv.py base.csv retry1.csv retry2.csv -o final.csv

典型流程(本次主实验 v1 + retry 融合):
    # 1. 用 backfill 后的 CSV 作为 base
    # 2. 跑 retry: --retry-errors expr_v1_with_padded.csv --padded-threshold 1
    # 3. (retry 完后)backfill_padded_from_stdout 把 retry 的 stdout 也补 padded
    #    或直接用 run_experiment 已经 surface 的 padded 列(reranker.py 已改)
    # 4. merge_csv base retry → final
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: F401


def load_csv(path: Path):
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return reader.fieldnames or [], list(reader)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("inputs", nargs="+", type=Path, help="CSV 文件,后面的覆盖前面的")
    p.add_argument("-o", "--output", type=Path, required=True)
    args = p.parse_args()

    merged: dict = {}
    schema: list = []
    for path in args.inputs:
        if not path.exists():
            raise SystemExit(f"not found: {path}")
        fns, rows = load_csv(path)
        if not schema:
            schema = list(fns)
        elif set(schema) != set(fns):
            extra = set(fns) - set(schema)
            missing = set(schema) - set(fns)
            if extra:
                print(f"  NOTE: {path.name} adds columns {sorted(extra)}; appending to schema")
                schema = schema + [c for c in fns if c not in schema]
            if missing:
                print(f"  NOTE: {path.name} missing columns {sorted(missing)}; will leave blank in those rows")
        n_new = n_replace = 0
        for r in rows:
            key = (r["query_id"], r["poison_set"], r["reranker_llm"])
            if key in merged:
                n_replace += 1
            else:
                n_new += 1
            merged[key] = r
        print(f"  {path.name}: {len(rows)} rows ({n_new} new, {n_replace} overrides)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=schema, extrasaction="ignore")
        writer.writeheader()
        for k in sorted(merged):
            writer.writerow(merged[k])

    print(f"\nWrote {len(merged)} merged rows to {args.output}")


if __name__ == "__main__":
    main()

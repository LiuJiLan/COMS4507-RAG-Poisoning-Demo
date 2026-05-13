"""
Merge multiple expr_*.csv files by (query_id, poison_set, reranker_llm) triplet;
later inputs override earlier ones (right-most CSV on the command line wins).
按 (query_id, poison_set, reranker_llm) 三元组融合多个 expr_*.csv,
后到的 row 覆盖前到的(命令行右边优先级高)。

Usage:
    python scripts/merge_csv.py base.csv retry1.csv retry2.csv -o final.csv

Typical flow (v1 + retry):
    1. Use the backfilled CSV as the base.
    2. Retry: --retry-errors expr_v1_with_padded.csv --padded-threshold 1
    3. After retry, either backfill again from the retry stdout, or rely on
       reranker.py emitting padded columns directly.
    4. merge_csv base retry → final.

典型流程(v1 主实验 + retry 融合):base(backfilled)→ retry 跑 padded anomaly →
merge 出 final。
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

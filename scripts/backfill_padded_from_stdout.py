"""
scripts/backfill_padded_from_stdout.py — one-shot 补救脚本

读取 PowerShell 跑批保存的 stdout log,提取每条 row 内 reranker
WARNING/ERROR 事件,反推 reranker_padded_clean / reranker_padded_poisoned
两列填回 CSV(主实验 2026-05-13 跑批后 reranker.py 还没 surface padded
信息,导致原 CSV 缺这两列;此脚本一次性补)。

Heuristic:
- 0 anomaly → (0, 0)
- 1 anomaly → 不能区分 clean / poisoned 一路,两列都填同 padded 值(保守)
- 2 anomaly → 第 1 个 = clean,第 2 个 = poisoned(pipeline 先 clean 后 poisoned)
- API fail (ERROR `LLM reranker failed: ...; falling back`) → padded = 10

用法:
    python scripts/backfill_padded_from_stdout.py \
        data/results/expr_20260513_172627.stdout.log \
        data/results/expr_20260513_172627.csv

输出 `expr_20260513_172627_with_padded.csv`,不动原 CSV。
"""
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config  # noqa: F401  pyarrow preload

ANCHOR_RE = re.compile(r"\[\s*(\d+)/\d+\]\s+q=(\S+?)\s+poison=(\S+?)\s+llm=(\S+?)\s")
WARN_PADDED_RE = re.compile(r"WARNING\s+reranker\s+\S+:\s+parser padded\s+(\d+)/(\d+)")
ERROR_FALLBACK_RE = re.compile(r"ERROR\s+LLM reranker failed:.*falling back")
ROW_END_RE = re.compile(r"^\s*→\s+(k2=|ERROR)")


def parse_stdout(path: Path) -> dict:
    """Return {(qid, poison_set, llm): [padded_value, ...]} keyed by row triplet."""
    rows: dict = {}
    cur_key = None
    cur_events: list = []

    def flush():
        if cur_key is not None:
            rows[cur_key] = cur_events.copy()

    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            m_anchor = ANCHOR_RE.search(line)
            if m_anchor:
                flush()
                _, qid, poison, llm = m_anchor.groups()
                cur_key = (qid, poison, llm)
                cur_events = []
            if cur_key is None:
                continue
            # Same anchor line may also contain an inline anomaly.
            m_warn = WARN_PADDED_RE.search(line)
            m_err = ERROR_FALLBACK_RE.search(line)
            if m_warn:
                cur_events.append(int(m_warn.group(1)))
            elif m_err:
                cur_events.append(10)
            # row end → reset (anchor 行的下一行通常是 → k2=...)
            if ROW_END_RE.match(line):
                flush()
                cur_key = None
                cur_events = []
    flush()
    return rows


def disambig(events: list) -> tuple:
    if not events:
        return 0, 0
    if len(events) == 1:
        return events[0], events[0]  # uncertain → conservative both-fill
    return events[0], events[1]


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: backfill_padded_from_stdout.py <stdout.log> <expr.csv>")
    stdout_path = Path(sys.argv[1])
    csv_path = Path(sys.argv[2])
    if not stdout_path.exists():
        raise SystemExit(f"not found: {stdout_path}")
    if not csv_path.exists():
        raise SystemExit(f"not found: {csv_path}")

    events_by_row = parse_stdout(stdout_path)
    print(f"Parsed {len(events_by_row)} row blocks from {stdout_path.name}")
    rows_with_events = sum(1 for v in events_by_row.values() if v)
    print(f"  rows with anomaly: {rows_with_events}")
    print(f"  total anomaly events: {sum(len(v) for v in events_by_row.values())}")

    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = list(reader)

    # Inject new columns just before elapsed_sec for readability (mirror current code order).
    new_cols = ["reranker_padded_clean", "reranker_padded_poisoned"]
    if "reranker_padded_clean" not in fieldnames:
        insert_at = fieldnames.index("elapsed_sec") if "elapsed_sec" in fieldnames else len(fieldnames)
        fieldnames = fieldnames[:insert_at] + new_cols + fieldnames[insert_at:]

    n_matched = 0
    n_unmatched = 0
    for r in rows:
        key = (r["query_id"], r["poison_set"], r["reranker_llm"])
        evts = events_by_row.get(key)
        if evts is None:
            n_unmatched += 1
            r["reranker_padded_clean"] = ""
            r["reranker_padded_poisoned"] = ""
            continue
        n_matched += 1
        pc, pp = disambig(evts)
        r["reranker_padded_clean"] = pc
        r["reranker_padded_poisoned"] = pp

    out_path = csv_path.with_name(csv_path.stem + "_with_padded.csv")
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nMatched {n_matched}/{len(rows)} CSV rows; unmatched: {n_unmatched}")
    n_clean_nonzero = sum(1 for r in rows if str(r.get("reranker_padded_clean")) not in ("", "0"))
    n_poisoned_nonzero = sum(1 for r in rows if str(r.get("reranker_padded_poisoned")) not in ("", "0"))
    n_either = sum(
        1 for r in rows
        if str(r.get("reranker_padded_clean")) not in ("", "0")
        or str(r.get("reranker_padded_poisoned")) not in ("", "0")
    )
    print(f"  rows with padded_clean > 0:    {n_clean_nonzero}")
    print(f"  rows with padded_poisoned > 0: {n_poisoned_nonzero}")
    print(f"  rows with any padded > 0:      {n_either}")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()

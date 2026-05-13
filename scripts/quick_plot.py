"""
scripts/quick_plot.py — ad-hoc 出一张图看实验结果

读 data/results/ 下最新一个 expr_*.csv,出每种 attack 的 ASR@k1 / ASR@k2 柱状对比,
保存为同名 .png。仅 demo / sanity 用,真正的 analysis notebook 实验跑完再写。

用法:
    python scripts/quick_plot.py                 # 用最新 CSV
    python scripts/quick_plot.py path/to/expr.csv
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config  # noqa: load pyarrow preload  # type: ignore
import pandas as pd
import matplotlib.pyplot as plt


def main():
    if len(sys.argv) > 1:
        csv_path = Path(sys.argv[1])
    else:
        results_dir = config.DATA_DIR / "results"
        csvs = sorted(results_dir.glob("expr_*.csv"), key=lambda p: p.stat().st_mtime)
        if not csvs:
            raise SystemExit(f"No expr_*.csv in {results_dir}")
        csv_path = csvs[-1]

    print(f"Plotting: {csv_path.name}")
    df = pd.read_csv(csv_path)
    df = df[df["error"].isna() | (df["error"] == "")]

    # ASR by poison_set
    grouped = df.groupby("poison_set").agg(
        k1_asr=("k1_attack_success", lambda s: s.eq(True).mean()),
        k2_asr=("k2_attack_success", lambda s: s.eq(True).mean()),
        n=("k1_attack_success", "size"),
    ).reset_index()
    grouped["attack"] = grouped["poison_set"].str.removeprefix("P_").str.replace("_", " ").str.title()
    grouped = grouped.sort_values("attack")

    n_llms = df["reranker_llm"].nunique()
    n_queries = df["query_id"].nunique()
    title = (f"ASR by attack — {csv_path.stem}\n"
             f"({n_queries} queries × {n_llms} LLM × {len(grouped)} attacks, n={len(df)} combos)")

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(grouped))
    w = 0.38
    ax.bar([i - w / 2 for i in x], grouped["k1_asr"], width=w, label="ASR @ k1 (dense retrieval)", color="#60a5fa")
    ax.bar([i + w / 2 for i in x], grouped["k2_asr"], width=w, label="ASR @ k2 (after rerank)", color="#f97316")
    ax.set_xticks(list(x))
    ax.set_xticklabels(grouped["attack"], rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Attack success rate")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)

    for i, row in grouped.iterrows():
        idx = list(grouped.index).index(i)
        ax.text(idx - w / 2, row["k1_asr"] + 0.02, f"{row['k1_asr']:.0%}", ha="center", fontsize=9)
        ax.text(idx + w / 2, row["k2_asr"] + 0.02, f"{row['k2_asr']:.0%}", ha="center", fontsize=9)

    fig.tight_layout()
    out_path = csv_path.with_suffix(".png")
    fig.savefig(out_path, dpi=120)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

"""
scripts/quick_plot.py — ad-hoc 出图看实验结果

读 data/results/ 下最新一个 expr_*.csv,出两张图:
  - expr_<ts>.png          — ASR by attack(4 LLM 加权)
  - expr_<ts>_by_llm.png   — ASR@k2 by LLM × attack(LLM 行为差异)

仅 demo / sanity 用,正式 analysis notebook 实验跑完再写。

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


def _attack_label(poison_set: str) -> str:
    return poison_set.removeprefix("P_").replace("_", " ").title()


def plot_by_attack(df: pd.DataFrame, csv_path: Path) -> Path:
    """ASR by attack — k1 vs k2 grouped bars,4 LLM 加权平均。"""
    grouped = df.groupby("poison_set").agg(
        k1_asr=("k1_attack_success", lambda s: s.eq(True).mean()),
        k2_asr=("k2_attack_success", lambda s: s.eq(True).mean()),
        n=("k1_attack_success", "size"),
    ).reset_index()
    grouped["attack"] = grouped["poison_set"].apply(_attack_label)
    grouped = grouped.sort_values("attack").reset_index(drop=True)

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
        ax.text(i - w / 2, row["k1_asr"] + 0.02, f"{row['k1_asr']:.0%}", ha="center", fontsize=9)
        ax.text(i + w / 2, row["k2_asr"] + 0.02, f"{row['k2_asr']:.0%}", ha="center", fontsize=9)

    fig.tight_layout()
    out_path = csv_path.with_suffix(".png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def plot_by_llm(df: pd.DataFrame, csv_path: Path) -> Path:
    """ASR@k2 by LLM × attack — 一组 attack 一簇,4 个 LLM bar。看 LLM 行为差异。"""
    grouped = df.groupby(["poison_set", "reranker_llm"]).agg(
        k2_asr=("k2_attack_success", lambda s: s.eq(True).mean()),
    ).reset_index()
    grouped["attack"] = grouped["poison_set"].apply(_attack_label)
    pivot = grouped.pivot(index="attack", columns="reranker_llm", values="k2_asr").sort_index()

    llms = list(pivot.columns)
    palette = {"claude": "#d97706", "gpt4o": "#10b981", "gemini": "#3b82f6", "llama": "#a855f7"}
    colors = [palette.get(l, "#888") for l in llms]

    fig, ax = plt.subplots(figsize=(12, 5))
    x = range(len(pivot))
    n_llm = len(llms)
    w = 0.8 / n_llm
    for j, llm in enumerate(llms):
        offsets = [i + (j - (n_llm - 1) / 2) * w for i in x]
        ax.bar(offsets, pivot[llm], width=w, label=llm, color=colors[j])
        for i, v in enumerate(pivot[llm]):
            if pd.notna(v):
                ax.text(offsets[i], v + 0.015, f"{v:.0%}", ha="center", fontsize=8)

    n_queries = df["query_id"].nunique()
    title = (f"ASR @ k2 by LLM × attack — {csv_path.stem}\n"
             f"({n_queries} queries × {n_llm} LLM × {len(pivot)} attacks, n={len(df)} combos)")
    ax.set_xticks(list(x))
    ax.set_xticklabels(pivot.index, rotation=15, ha="right")
    ax.set_ylim(0, 1.10)
    ax.set_ylabel("Attack success rate (after rerank)")
    ax.set_title(title, fontsize=11)
    ax.legend(loc="lower right", ncol=n_llm, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    out_path = csv_path.with_name(csv_path.stem + "_by_llm.png")
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


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

    p1 = plot_by_attack(df, csv_path)
    print(f"Saved: {p1.name}")
    p2 = plot_by_llm(df, csv_path)
    print(f"Saved: {p2.name}")


if __name__ == "__main__":
    main()

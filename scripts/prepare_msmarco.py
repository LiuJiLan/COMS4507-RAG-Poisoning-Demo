"""
Pull MS MARCO v2.1 from HuggingFace and sample 5000 passages as background corpus.
从 HuggingFace 拉 MS MARCO v2.1,抽样 5000 篇 passages 作为背景集合(ADJ-001)。

Usage:
    python scripts/prepare_msmarco.py

Output:
    data/corpus_static/msmarco_background.json (~2-3 MB, ~5000 Document records)

Notes:
- seed=42 for reproducibility (so course markers reproducing the experiment
  get the same background set).
- MS MARCO v2.1 train split is ~3-5 GB on HF; first download is bandwidth-bound.
- HF caches under ~/.cache/huggingface; second run reads from cache and is fast.
- No topic filtering — messy content actually supports the "real RAG corpora
  are messy" framing.

说明:seed=42 可复现;首次下载 3-5 GB,后续走 HF 缓存;不做主题 filter,杂乱内容
反而支撑"现实 RAG corpus 就是混杂的"的 framing。
"""
import sys
import json
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import config

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("prepare_msmarco")

SEED = 42
N_TARGET = 5000
# Oversample to absorb empty rows; MS MARCO has almost none, so 200 buffer is plenty.
# 多抽一些,跳过空 passage 后仍能凑足 5000;MS MARCO 几乎没空 row,200 buffer 绰绰有余。
N_OVERSAMPLE = N_TARGET + 200


def _first_passage(passages_field) -> str:
    """
    Return the first non-empty passage_text from a v2.1 passages field.
    从 v2.1 的 passages 字段拿第一条非空 passage_text。
    """
    if not isinstance(passages_field, dict):
        return ""
    texts = passages_field.get("passage_text") or []
    for t in texts:
        if t and t.strip():
            return t.strip()
    return ""


def _make_title(content: str, n_words: int = 10) -> str:
    """
    Use the first n words as a pseudo title.
    用前 n 个词作为伪 title。
    """
    words = content.split()
    return " ".join(words[:n_words]) if words else "[untitled]"


def main() -> int:
    out_path = config.BACKGROUND_CORPUS_FILE
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if out_path.exists():
        logger.warning(f"{out_path} already exists. Re-running will overwrite it.")

    logger.info("Loading MS MARCO v2.1 train split via HuggingFace datasets...")
    from datasets import load_dataset
    ds = load_dataset("ms_marco", "v2.1", split="train")
    logger.info(f"  total rows in train split: {len(ds)}")

    logger.info(f"Shuffling with seed={SEED} and selecting {N_OVERSAMPLE} rows...")
    sample = ds.shuffle(seed=SEED).select(range(N_OVERSAMPLE))

    logger.info("Adapting to project Document schema...")
    docs = []
    skipped_empty = 0
    for row in sample:
        if len(docs) >= N_TARGET:
            break
        content = _first_passage(row.get("passages"))
        if not content:
            skipped_empty += 1
            continue
        doc_id = f"msmarco_{len(docs) + 1:05d}"
        docs.append({
            "doc_id": doc_id,
            "title": _make_title(content),
            "content": content,
            "source": "msmarco",
            "topic": "background",
            "url": "",
        })

    if len(docs) < N_TARGET:
        logger.error(f"Only got {len(docs)} valid docs (target {N_TARGET}). "
                     f"Increase N_OVERSAMPLE.")
        return 1

    logger.info(f"Adapted {len(docs)} docs (skipped {skipped_empty} empty rows).")

    logger.info(f"Writing to: {out_path}")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(docs, f, ensure_ascii=False, indent=2)

    size_mb = out_path.stat().st_size / 1024 / 1024
    logger.info(f"Done. File size: {size_mb:.2f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())

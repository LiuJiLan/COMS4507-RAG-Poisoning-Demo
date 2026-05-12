"""
ADJ-001: 从 HuggingFace 拉 MS MARCO v2.1,抽样 5000 篇 passages 作为背景集合。

用法:
    python scripts/prepare_msmarco.py

输出:
    data/corpus_static/msmarco_background.json (~2-3 MB,~5000 条 Document 记录)

说明:
- seed=42 保证可复现 (UQ 老师/助教复现实验时拿到同一组背景文档)。
- MS MARCO v2.1 train split 在 HF 上约 3-5 GB,首次下载耗时取决于带宽。
- HF 缓存默认在 ~/.cache/huggingface,二次运行从缓存读,秒级。
- 不做主题 filter —— 杂乱内容反而支撑 "现实 RAG corpus 就是混杂的" 的 framing。
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
# 多抽一些,跳过空 passage 后仍能凑足 5000。MS MARCO 几乎没空 row,200 buffer 绰绰有余。
N_OVERSAMPLE = N_TARGET + 200


def _first_passage(passages_field) -> str:
    """从 v2.1 的 passages 字段拿第一条非空 passage_text。"""
    if not isinstance(passages_field, dict):
        return ""
    texts = passages_field.get("passage_text") or []
    for t in texts:
        if t and t.strip():
            return t.strip()
    return ""


def _make_title(content: str, n_words: int = 10) -> str:
    """前 n 个词作为伪 title。"""
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

"""
Offline FAISS index build script.
离线 FAISS 索引构建脚本。

Run this once before launching the app so the FAISS index is cached to disk;
the app then loads it in seconds on subsequent starts.
启动 app 前跑一次,把 FAISS 索引缓存到磁盘,之后 app 启动秒级从 cache 加载。

Usage:
    python scripts/build_index.py
"""
import sys
from pathlib import Path

# Let `python scripts/build_index.py` import top-level project modules.
# 让 `python scripts/build_index.py` 能 import 项目模块。
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import config
from src.corpus import load_corpus
from src.embedder import Embedder
from src.retriever import FAISSRetriever


def main():
    print(f"Loading BASE corpus from: {config.BASE_CORPUS_FILE}")
    base_docs = load_corpus(config.BASE_CORPUS_FILE)
    print(f"  base docs: {len(base_docs)}")

    print(f"Loading BACKGROUND corpus from: {config.BACKGROUND_CORPUS_FILE}")
    background_docs = load_corpus(config.BACKGROUND_CORPUS_FILE)
    print(f"  background docs: {len(background_docs)}")

    all_docs = base_docs + background_docs
    print(f"Combined corpus: {len(all_docs)} documents")

    print(f"Loading embedder: {config.EMBEDDING_MODEL} on {config.EMBEDDING_DEVICE}")
    embedder = Embedder(model_name=config.EMBEDDING_MODEL,
                        device=config.EMBEDDING_DEVICE)

    print("Building FAISS index...")
    retriever = FAISSRetriever(embedder=embedder)
    retriever.build_from_documents(all_docs, show_progress=True)

    print(f"Saving to cache:\n  {config.FAISS_CACHE}\n  {config.DOCS_CACHE}")
    retriever.save(config.FAISS_CACHE, config.DOCS_CACHE)

    print("Done!")


if __name__ == "__main__":
    main()

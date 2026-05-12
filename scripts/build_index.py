"""
离线建索引脚本。在跑 app 之前先跑一次这个，把 FAISS 索引缓存到磁盘。

用法：
    python scripts/build_index.py

之后启动 app 会自动从 cache 加载，秒级启动。
"""
import sys
from pathlib import Path

# 让 scripts 能 import 项目模块
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

import config
from src.corpus import load_corpus
from src.embedder import Embedder
from src.retriever import FAISSRetriever


def main():
    print(f"Loading corpus from: {config.DEFAULT_CORPUS_FILE}")
    docs = load_corpus(config.DEFAULT_CORPUS_FILE)
    print(f"Loaded {len(docs)} documents")

    print(f"Loading embedder: {config.EMBEDDING_MODEL} on {config.EMBEDDING_DEVICE}")
    embedder = Embedder(model_name=config.EMBEDDING_MODEL,
                        device=config.EMBEDDING_DEVICE)

    print("Building FAISS index...")
    retriever = FAISSRetriever(embedder=embedder)
    retriever.build_from_documents(docs, show_progress=True)

    print(f"Saving to cache:\n  {config.FAISS_CACHE}\n  {config.DOCS_CACHE}")
    retriever.save(config.FAISS_CACHE, config.DOCS_CACHE)

    print("Done!")


if __name__ == "__main__":
    main()

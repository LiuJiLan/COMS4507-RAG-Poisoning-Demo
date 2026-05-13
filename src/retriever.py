"""
FAISS-based dense retrieval.
基于 FAISS 的密集向量检索。

Design notes:
- IndexFlatIP with L2-normalised vectors gives cosine similarity.
- Persists to disk (save/load) so app startup is fast on warm runs.
- Supports run-time poison injection / removal via a token handle.
- Owns the doc_id <-> faiss-position mapping (parallel array).

设计要点:cosine 用 IndexFlatIP + L2 normalize;支持持久化;支持运行时
poison 注入与 token 化清理;内部用平行数组维护 doc_id ↔ faiss 位置。
"""
from pathlib import Path
from typing import List, Optional
import numpy as np
import json
import faiss
import logging

from .corpus import Document
from .embedder import Embedder

logger = logging.getLogger(__name__)


class RetrievalResult:
    """
    A single retrieval hit (document + score + 1-indexed rank).
    单条检索结果(文档 + 分数 + 1-indexed rank)。
    """
    def __init__(self, doc: Document, score: float, rank: int):
        self.doc = doc
        self.score = score
        self.rank = rank  # 1-indexed

    def __repr__(self):
        marker = " ☣" if self.doc.is_poison else ""
        return f"Rank {self.rank}: {self.doc.doc_id}{marker} (score={self.score:.4f})"


class FAISSRetriever:
    """
    Dense retrieval via FAISS.
    用 FAISS 做密集检索。

    Usage:
        retriever = FAISSRetriever(embedder)
        retriever.build_from_documents(docs)
        results = retriever.search("my query", k=5)

        # Inject poison temporarily.
        token = retriever.inject_poison(poison_docs)
        results_poisoned = retriever.search("my query", k=5)
        retriever.remove_poison(token)
        # Index is back to the original state.
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index: Optional[faiss.Index] = None
        # Parallel array: faiss position -> Document.
        # 平行数组:faiss index 位置 -> Document。
        self.documents: List[Document] = []
        # Poison injection bookkeeping: token -> list of faiss indices added by this token.
        # poison 注入记账:token -> 该次注入的 faiss 索引列表。
        self._poison_tokens: dict[int, List[int]] = {}
        self._next_token = 1

    # ------------------------------------------------------------------
    # Build / persist
    # 构建 / 持久化
    # ------------------------------------------------------------------
    def build_from_documents(self, docs: List[Document], show_progress: bool = True):
        """
        Build the index from a list of documents.
        从文档列表构建索引。
        """
        if not docs:
            raise ValueError("No documents to build index from")

        texts = [d.text_for_embedding for d in docs]
        logger.info(f"Embedding {len(texts)} documents...")
        vectors = self.embedder.encode(texts, show_progress=show_progress)

        # Normalise for cosine similarity.
        # L2 归一化,把内积变成 cosine。
        faiss.normalize_L2(vectors)

        # IndexFlatIP — exact search, fast enough for <100k docs.
        # IndexFlatIP 精确搜索,<100k 文档量足够快。
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents = list(docs)
        logger.info(f"Index built: {self.index.ntotal} vectors")

    def save(self, faiss_path: Path, docs_path: Path):
        """
        Save the index and the document list to disk.
        保存索引和文档列表。
        """
        faiss_path = Path(faiss_path)
        docs_path = Path(docs_path)
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(faiss_path))
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump([d.to_dict() for d in self.documents], f,
                      ensure_ascii=False, indent=2)
        logger.info(f"Saved index to {faiss_path} and docs to {docs_path}")

    def load(self, faiss_path: Path, docs_path: Path):
        """
        Load index + documents from disk.
        从磁盘加载索引与文档。
        """
        from .corpus import load_corpus
        faiss_path = Path(faiss_path)
        docs_path = Path(docs_path)
        if not faiss_path.exists() or not docs_path.exists():
            raise FileNotFoundError(f"Cache files missing: {faiss_path} or {docs_path}")
        self.index = faiss.read_index(str(faiss_path))
        self.documents = load_corpus(docs_path)
        logger.info(f"Loaded index from cache: {self.index.ntotal} vectors")

    # ------------------------------------------------------------------
    # Search
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """
        Retrieve top-k documents for the given query.
        检索 query 对应的 top-k 文档。

        Returns:
            List[RetrievalResult] sorted ascending by rank.
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_from_documents() first.")
        if self.index.ntotal == 0:
            return []

        # Embed and normalise the query vector.
        # query 编码 + L2 归一化。
        q_vec = self.embedder.encode(query).reshape(1, -1)
        faiss.normalize_L2(q_vec)

        k_actual = min(k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k_actual)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(self.documents):
                continue  # safety bound check / 边界保护
            results.append(RetrievalResult(
                doc=self.documents[idx],
                score=float(score),
                rank=rank,
            ))
        return results

    # ------------------------------------------------------------------
    # Poison injection (runtime, incremental)
    # poison 注入(运行时增量)
    # ------------------------------------------------------------------
    def inject_poison(self, poison_docs: List[Document]) -> int:
        """
        Temporarily add poison documents to the index. Returns a token used to remove them later.
        临时加 poison 文档,返回 token 用于后续移除。
        """
        if not poison_docs:
            return -1
        if self.index is None:
            raise RuntimeError("Index not built yet")

        texts = [d.text_for_embedding for d in poison_docs]
        vectors = self.embedder.encode(texts)
        faiss.normalize_L2(vectors)

        # Mark as poison.
        # 打 poison 标记。
        for d in poison_docs:
            d.is_poison = True

        start_idx = self.index.ntotal
        self.index.add(vectors)
        new_indices = list(range(start_idx, start_idx + len(poison_docs)))
        self.documents.extend(poison_docs)

        token = self._next_token
        self._next_token += 1
        self._poison_tokens[token] = new_indices
        logger.info(f"Injected {len(poison_docs)} poison docs, token={token}")
        return token

    def remove_poison(self, token: int):
        """
        Remove a previously-injected poison batch identified by token.
        按 token 移除之前注入的 poison 批次。

        IndexFlatIP cannot delete in place, so we rebuild the index from the
        remaining (non-this-token) documents. For <1k docs this is sub-second.

        IndexFlatIP 不支持原地删除,这里直接用剩余文档重建索引;<1k 文档秒级完成。
        """
        if token not in self._poison_tokens:
            logger.warning(f"Unknown poison token: {token}")
            return

        indices_to_remove = set(self._poison_tokens.pop(token))

        # Keep everything except the docs added by this token.
        # 保留所有非本次注入的文档。
        kept_docs = [d for i, d in enumerate(self.documents)
                     if i not in indices_to_remove]

        # Note: kept_docs may still contain poison from other tokens; their
        # vectors will be recomputed in build_from_documents.
        # 注意:kept_docs 可能含其他 token 的 poison,它们的向量会在 build 时重算。
        old_n = self.index.ntotal
        self.build_from_documents(kept_docs, show_progress=False)
        logger.info(f"Removed {len(indices_to_remove)} poison docs "
                    f"(was {old_n}, now {self.index.ntotal})")

    def reset_all_poison(self):
        """
        Drop every poison document (used by the UI "reset" button).
        清掉全部 poison(UI "回到初始状态" 按钮用)。
        """
        clean_docs = [d for d in self.documents if not d.is_poison]
        if len(clean_docs) < len(self.documents):
            self.build_from_documents(clean_docs, show_progress=False)
            self._poison_tokens.clear()

    # ------------------------------------------------------------------
    # Stats
    # 元信息
    # ------------------------------------------------------------------
    @property
    def n_documents(self) -> int:
        return len(self.documents)

    @property
    def n_poison(self) -> int:
        return sum(1 for d in self.documents if d.is_poison)

    @property
    def n_clean(self) -> int:
        return self.n_documents - self.n_poison

"""
FAISS 向量检索。

设计要点：
- 用 IndexFlatIP（cosine similarity，需要先 normalize 向量）
- 支持持久化（save/load）
- 支持运行时增量添加文档（用 poison 注入）
- 内部维护 doc_id <-> faiss_idx 的映射
"""
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import json
import faiss
import logging

from .corpus import Document
from .embedder import Embedder

logger = logging.getLogger(__name__)


class RetrievalResult:
    """单个检索结果"""
    def __init__(self, doc: Document, score: float, rank: int):
        self.doc = doc
        self.score = score
        self.rank = rank  # 1-indexed

    def __repr__(self):
        marker = " ☣" if self.doc.is_poison else ""
        return f"Rank {self.rank}: {self.doc.doc_id}{marker} (score={self.score:.4f})"


class FAISSRetriever:
    """
    用 FAISS 做 dense retrieval。
    
    用法：
        retriever = FAISSRetriever(embedder)
        retriever.build_from_documents(docs)
        results = retriever.search("my query", k=5)
        # results 是 List[RetrievalResult]
        
        # 临时注入 poison
        token = retriever.inject_poison(poison_docs)
        results_poisoned = retriever.search("my query", k=5)
        retriever.remove_poison(token)
        # 索引回到初始状态
    """

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.index: Optional[faiss.Index] = None
        # 平行数组，index_position -> Document
        self.documents: List[Document] = []
        # 用于 poison 注入：token -> list of indices added by this token
        self._poison_tokens: dict[int, List[int]] = {}
        self._next_token = 1

    # ------------------------------------------------------------------
    # 构建 / 持久化
    # ------------------------------------------------------------------
    def build_from_documents(self, docs: List[Document], show_progress: bool = True):
        """从一组文档构建索引"""
        if not docs:
            raise ValueError("No documents to build index from")

        texts = [d.text_for_embedding for d in docs]
        logger.info(f"Embedding {len(texts)} documents...")
        vectors = self.embedder.encode(texts, show_progress=show_progress)

        # Normalize for cosine similarity
        faiss.normalize_L2(vectors)

        # Create flat index (exact search; fast enough for <100k docs)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.documents = list(docs)
        logger.info(f"Index built: {self.index.ntotal} vectors")

    def save(self, faiss_path: Path, docs_path: Path):
        """保存索引和文档列表"""
        faiss_path = Path(faiss_path)
        docs_path = Path(docs_path)
        faiss_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(faiss_path))
        with open(docs_path, "w", encoding="utf-8") as f:
            json.dump([d.to_dict() for d in self.documents], f,
                      ensure_ascii=False, indent=2)
        logger.info(f"Saved index to {faiss_path} and docs to {docs_path}")

    def load(self, faiss_path: Path, docs_path: Path):
        """从磁盘加载索引和文档"""
        from .corpus import load_corpus
        faiss_path = Path(faiss_path)
        docs_path = Path(docs_path)
        if not faiss_path.exists() or not docs_path.exists():
            raise FileNotFoundError(f"Cache files missing: {faiss_path} or {docs_path}")
        self.index = faiss.read_index(str(faiss_path))
        self.documents = load_corpus(docs_path)
        logger.info(f"Loaded index from cache: {self.index.ntotal} vectors")

    # ------------------------------------------------------------------
    # 检索
    # ------------------------------------------------------------------
    def search(self, query: str, k: int = 10) -> List[RetrievalResult]:
        """
        检索 top-k 文档。
        
        Returns:
            List[RetrievalResult]，按 rank 升序排列
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_from_documents() first.")
        if self.index.ntotal == 0:
            return []

        # Embed query
        q_vec = self.embedder.encode(query).reshape(1, -1)
        faiss.normalize_L2(q_vec)

        # Search
        k_actual = min(k, self.index.ntotal)
        scores, indices = self.index.search(q_vec, k_actual)

        results = []
        for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
            if idx < 0 or idx >= len(self.documents):
                continue  # 安全检查
            results.append(RetrievalResult(
                doc=self.documents[idx],
                score=float(score),
                rank=rank,
            ))
        return results

    # ------------------------------------------------------------------
    # Poison 注入（运行时增量）
    # ------------------------------------------------------------------
    def inject_poison(self, poison_docs: List[Document]) -> int:
        """
        临时添加 poison 文档到索引。返回 token 用于后续移除。
        """
        if not poison_docs:
            return -1
        if self.index is None:
            raise RuntimeError("Index not built yet")

        texts = [d.text_for_embedding for d in poison_docs]
        vectors = self.embedder.encode(texts)
        faiss.normalize_L2(vectors)

        # 标记 poison
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
        移除之前注入的 poison。
        
        注意：FAISS 的 IndexFlatIP 不支持 in-place 删除，
        我们的策略是"重建索引但保留非 poison 文档"。
        因为只删除少量 poison，重建很快（<1秒）。
        """
        if token not in self._poison_tokens:
            logger.warning(f"Unknown poison token: {token}")
            return

        indices_to_remove = set(self._poison_tokens.pop(token))

        # 留下非这次注入的文档
        kept_docs = [d for i, d in enumerate(self.documents)
                     if i not in indices_to_remove]

        # 重建索引
        # 注意：留下的 documents 可能还包含其他 poison（不同 token）
        # 它们的 embedding 在重新 build 时会重新计算
        # 这里偷个懒：直接全量重建
        # 性能：1000 文档 ~5 秒（GPU 上 <1 秒），完全可接受
        old_n = self.index.ntotal
        self.build_from_documents(kept_docs, show_progress=False)
        logger.info(f"Removed {len(indices_to_remove)} poison docs "
                    f"(was {old_n}, now {self.index.ntotal})")

    def reset_all_poison(self):
        """清除所有 poison（用于"回到初始状态"按钮）"""
        clean_docs = [d for d in self.documents if not d.is_poison]
        if len(clean_docs) < len(self.documents):
            self.build_from_documents(clean_docs, show_progress=False)
            self._poison_tokens.clear()

    # ------------------------------------------------------------------
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

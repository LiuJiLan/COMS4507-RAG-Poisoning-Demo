"""
Pipeline assembly — wires every component into one RAG poisoning experiment flow.
Pipeline 总装配,把所有组件拼成一个 RAG poisoning 实验流程。

The key method is run_experiment(): take a query + poison docs, return a full
clean-vs-poisoned comparison across dense retrieval, LLM reranking, and (optionally)
LLM generation.

最重要的方法是 run_experiment():输入 query 和 poison docs,返回 dense 检索 +
LLM 重排 +(可选)LLM 生成三阶段的 clean vs poisoned 完整对比。
"""
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
import logging

from .corpus import Document, load_corpus
from .embedder import Embedder
from .retriever import FAISSRetriever, RetrievalResult
from .reranker import LLMReranker
from .generator import LLMGenerator
from .llm_clients import LLMClient, StubLLMClient
from .evaluator import AttackMetrics, compare_rankings

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """
    Output of one query going through the full pipeline.
    单次 query 跑完整 pipeline 的输出。
    """
    query: str
    # k_1 stage (dense retriever)
    # k_1 阶段:dense retriever
    top_k1_clean: List[RetrievalResult]
    top_k1_poisoned: List[RetrievalResult]
    metrics_k1: AttackMetrics
    # k_2 stage (LLM reranker)
    # k_2 阶段:LLM reranker
    top_k2_clean: List[RetrievalResult]
    top_k2_poisoned: List[RetrievalResult]
    metrics_k2: AttackMetrics
    # k_2 reranker completeness telemetry: how many positions the parser
    # back-filled in dense order. 0 = LLM emitted full ranking; 1..N-1 =
    # under-output; N = API call failed (full dense fallback).
    # For experiment post-processing; the UI does not need to care.
    # k_2 reranker 完整性 telemetry:parser 按 dense 原序补齐的位置数。
    # 0 = LLM 完整;1..N-1 = under-output;N = API fail 完全 fallback。
    # 给主实验后处理用,UI 不必关心。
    reranker_padded_clean: int = 0
    reranker_padded_poisoned: int = 0
    # Generator stage
    # generator 阶段
    answer_clean: str = ""
    answer_poisoned: str = ""


class RAGPipeline:
    """
    Full RAG poisoning experiment pipeline.
    完整的 RAG poisoning 实验 pipeline。

    Usage:
        pipeline = RAGPipeline()
        pipeline.initialize(corpus_docs)  # one-off
        result = pipeline.run_experiment(
            query="...",
            poison_docs=[Document(...), ...],
        )
    """

    def __init__(self,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 embedding_device: str = "cpu",
                 reranker_client: Optional[LLMClient] = None,
                 generator_client: Optional[LLMClient] = None,
                 top_k_1: int = 10,
                 top_k_2: int = 5):
        self.embedder = Embedder(model_name=embedding_model, device=embedding_device)
        self.retriever = FAISSRetriever(embedder=self.embedder)
        # Default to stub clients; caller can pass real clients when available.
        # 默认 stub,caller 可注入真实 client。
        self.reranker = LLMReranker(llm_client=reranker_client or StubLLMClient())
        self.generator = LLMGenerator(llm_client=generator_client or StubLLMClient())
        self.top_k_1 = top_k_1
        self.top_k_2 = top_k_2

    def initialize(self, docs: List[Document]):
        """
        Build the index from a document list.
        从文档列表构建索引。
        """
        self.retriever.build_from_documents(docs)

    def load_cached_index(self, faiss_path: Path, docs_path: Path):
        """
        Load a pre-built index from disk.
        从磁盘加载已建好的索引。
        """
        self.retriever.load(faiss_path, docs_path)

    def run_experiment(self, query: str,
                       poison_docs: List[Document],
                       include_generator: bool = False) -> PipelineResult:
        """
        Run one full clean-vs-poisoned comparison.
        跑一次完整的 clean vs poisoned 对比实验。

        Steps:
        1. Clean retrieval (k_1)
        2. Clean rerank (k_2)
        3. (Optional) clean generate
        4. Inject poison
        5. Poisoned retrieval (k_1)
        6. Poisoned rerank (k_2)
        7. (Optional) poisoned generate
        8. Remove poison (cleanup, even on error)

        步骤:1) clean retrieve  2) clean rerank  3) (可选) clean generate
              4) 注入 poison  5) poisoned retrieve  6) poisoned rerank
              7) (可选) poisoned generate  8) 清理 poison(异常也清)。
        """
        # ----- Clean side -----
        top_k1_clean = self.retriever.search(query, k=self.top_k_1)
        top_k2_clean, padded_clean = self.reranker.rerank(query, top_k1_clean, top_k=self.top_k_2)
        answer_clean = ""
        if include_generator:
            answer_clean = self.generator.generate(query, top_k2_clean)

        # ----- Poisoned side -----
        token = self.retriever.inject_poison(poison_docs)
        try:
            top_k1_poisoned = self.retriever.search(query, k=self.top_k_1)
            top_k2_poisoned, padded_poisoned = self.reranker.rerank(
                query, top_k1_poisoned, top_k=self.top_k_2,
            )
            answer_poisoned = ""
            if include_generator:
                answer_poisoned = self.generator.generate(query, top_k2_poisoned)
        finally:
            # Always clean up the poison so the index returns to a known state.
            # 无论如何都清理 poison,保证索引干净。
            self.retriever.remove_poison(token)

        # ----- Metrics -----
        metrics_k1 = compare_rankings(top_k1_clean, top_k1_poisoned)
        metrics_k2 = compare_rankings(top_k2_clean, top_k2_poisoned)

        return PipelineResult(
            query=query,
            top_k1_clean=top_k1_clean,
            top_k1_poisoned=top_k1_poisoned,
            metrics_k1=metrics_k1,
            top_k2_clean=top_k2_clean,
            top_k2_poisoned=top_k2_poisoned,
            metrics_k2=metrics_k2,
            reranker_padded_clean=padded_clean,
            reranker_padded_poisoned=padded_poisoned,
            answer_clean=answer_clean,
            answer_poisoned=answer_poisoned,
        )

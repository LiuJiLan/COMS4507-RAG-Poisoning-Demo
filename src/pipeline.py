"""
Pipeline 总装配。把所有组件拼成一个 RAG poisoning 实验流程。

最重要的方法：run_experiment()
输入 query 和 poison docs，输出"clean vs poisoned"的完整对比。
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
    """单次 query 跑完整 pipeline 的输出"""
    query: str
    # k_1 阶段（dense retriever）
    top_k1_clean: List[RetrievalResult]
    top_k1_poisoned: List[RetrievalResult]
    metrics_k1: AttackMetrics
    # k_2 阶段（LLM reranker）
    top_k2_clean: List[RetrievalResult]
    top_k2_poisoned: List[RetrievalResult]
    metrics_k2: AttackMetrics
    # generator 阶段
    answer_clean: str = ""
    answer_poisoned: str = ""


class RAGPipeline:
    """
    完整的 RAG poisoning 实验 pipeline。
    
    用法：
        pipeline = RAGPipeline()
        pipeline.initialize(corpus_docs)  # 一次性
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
        # 默认用 stub，明天接真实 API 后传入真实 client
        self.reranker = LLMReranker(llm_client=reranker_client or StubLLMClient())
        self.generator = LLMGenerator(llm_client=generator_client or StubLLMClient())
        self.top_k_1 = top_k_1
        self.top_k_2 = top_k_2

    def initialize(self, docs: List[Document]):
        """从文档列表构建索引"""
        self.retriever.build_from_documents(docs)

    def load_cached_index(self, faiss_path: Path, docs_path: Path):
        """从磁盘加载已建好的索引"""
        self.retriever.load(faiss_path, docs_path)

    def run_experiment(self, query: str,
                       poison_docs: List[Document],
                       include_generator: bool = False) -> PipelineResult:
        """
        跑一次完整的对比实验。
        
        步骤：
        1. clean retrieval（k_1）
        2. clean rerank（k_2）
        3. （可选）clean generate
        4. 注入 poison
        5. poisoned retrieval（k_1）
        6. poisoned rerank（k_2）
        7. （可选）poisoned generate
        8. 移除 poison（清理）
        """
        # ----- Clean side -----
        top_k1_clean = self.retriever.search(query, k=self.top_k_1)
        top_k2_clean = self.reranker.rerank(query, top_k1_clean, top_k=self.top_k_2)
        answer_clean = ""
        if include_generator:
            answer_clean = self.generator.generate(query, top_k2_clean)

        # ----- Poisoned side -----
        token = self.retriever.inject_poison(poison_docs)
        try:
            top_k1_poisoned = self.retriever.search(query, k=self.top_k_1)
            top_k2_poisoned = self.reranker.rerank(query, top_k1_poisoned,
                                                    top_k=self.top_k_2)
            answer_poisoned = ""
            if include_generator:
                answer_poisoned = self.generator.generate(query, top_k2_poisoned)
        finally:
            # 无论如何都要清理 poison，保证索引干净
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
            answer_clean=answer_clean,
            answer_poisoned=answer_poisoned,
        )

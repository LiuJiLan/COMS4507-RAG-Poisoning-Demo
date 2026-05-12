"""
LLM-as-reranker。

输入：query + top-k_1 文档列表
输出：重排后的 top-k_2 文档列表（k_2 ≤ k_1）

Prompt 设计：用编号列出所有文档，让 LLM 返回新的排名顺序。
这种"listwise"方式比"pointwise scoring"更稳定。
"""
from typing import List
import logging
import re

from .retriever import RetrievalResult
from .llm_clients import LLMClient

logger = logging.getLogger(__name__)


# Prompt 模板：listwise reranking
RERANK_PROMPT_TEMPLATE = """You are a search ranking assistant. Given a query and a list of documents, rank them by relevance to the query.

Query: {query}

Documents:
{document_block}

Instructions:
- Rank the documents from most relevant to least relevant.
- Return ONLY a comma-separated list of document numbers in your preferred order.
- Example output format: 3,1,5,2,4
- Do not include explanations or any other text.

Ranking:"""


def _format_documents(docs: List[RetrievalResult]) -> str:
    """把文档列表格式化成 prompt 里的 block"""
    lines = []
    for i, r in enumerate(docs, start=1):
        # 截断 content 防止 prompt 过长（保留前 500 字符）
        snippet = r.doc.content[:500]
        if len(r.doc.content) > 500:
            snippet += "..."
        lines.append(f"[{i}] Title: {r.doc.title}\n    {snippet}")
    return "\n\n".join(lines)


def _parse_ranking(response: str, n: int) -> List[int]:
    """
    从 LLM 输出里解析出排名。
    
    期望格式："3,1,5,2,4"，但要容错处理各种"AI 风格"输出。
    
    Args:
        response: LLM 原始输出
        n: 期望的文档数
    
    Returns:
        长度为 n 的 list，每个元素是 1..n 的 unique index
    """
    # 提取所有数字
    numbers = re.findall(r"\d+", response)
    seen = set()
    ranking = []
    for s in numbers:
        try:
            idx = int(s)
        except ValueError:
            continue
        if 1 <= idx <= n and idx not in seen:
            ranking.append(idx)
            seen.add(idx)
        if len(ranking) == n:
            break

    # 如果 LLM 漏了某些 index，补在末尾（保持原顺序）
    for i in range(1, n + 1):
        if i not in seen:
            ranking.append(i)

    return ranking[:n]


class LLMReranker:
    """
    用 LLM 对 dense retriever 的 top-k_1 进行重排。
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def rerank(self, query: str, candidates: List[RetrievalResult],
               top_k: int = 5) -> List[RetrievalResult]:
        """
        重排 candidates，返回 top_k。
        
        Args:
            query: 用户 query
            candidates: 来自 dense retriever 的 top-k_1 文档
            top_k: 要返回的 top_k_2 数量
        
        Returns:
            List[RetrievalResult]，按 LLM 的排序排列，rank 字段已更新为 1..k_2
        """
        if not candidates:
            return []
        if len(candidates) == 1:
            # 只有一个文档，没必要重排
            candidates[0].rank = 1
            return candidates[:top_k]

        prompt = RERANK_PROMPT_TEMPLATE.format(
            query=query,
            document_block=_format_documents(candidates),
        )

        try:
            response = self.llm.complete(prompt, max_tokens=200, temperature=0.0)
        except Exception as e:
            logger.error(f"LLM reranker failed: {e}; falling back to original order")
            response = ",".join(str(i) for i in range(1, len(candidates) + 1))

        ranking = _parse_ranking(response, len(candidates))

        # 把 1-indexed 排名转回 RetrievalResult
        reranked = []
        for new_rank, orig_idx in enumerate(ranking, start=1):
            r = candidates[orig_idx - 1]
            # 创建一个新的 RetrievalResult，避免修改原对象
            new_r = RetrievalResult(doc=r.doc, score=r.score, rank=new_rank)
            reranked.append(new_r)

        return reranked[:top_k]

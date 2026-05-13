"""
LLM-as-reranker。

输入：query + top-k_1 文档列表
输出：重排后的 top-k_2 文档列表（k_2 ≤ k_1）

Prompt 设计：用编号列出所有文档，让 LLM 返回新的排名顺序。
这种"listwise"方式比"pointwise scoring"更稳定。
"""
from typing import List, Tuple
import logging
import os
import re
import sys

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


def _parse_ranking(response: str, n: int) -> tuple[List[int], int]:
    """
    从 LLM 输出里解析出排名。

    期望格式："3,1,5,2,4"，但要容错处理各种"AI 风格"输出。

    Args:
        response: LLM 原始输出
        n: 期望的文档数

    Returns:
        (ranking, missing_count) 元组:
          - ranking: 长度为 n 的 list,每个元素是 1..n 的 unique index
          - missing_count: LLM 输出里**没**给出的 valid index 数(被 parser 按原序补齐的数量)。
            > 0 意味着 LLM "偷懒"或 silent fallback —— smoketest 2026-05-12 观察到
            GPT-4o-mini / Llama 3.3 在 listwise rerank 时只输出 top-5,后 5 位由
            parser 按 dense retriever 原序补齐。caller 应在 missing > 0 时 logger.warning。
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

    missing = n - len(ranking)

    # 如果 LLM 漏了某些 index，补在末尾（保持原顺序）
    for i in range(1, n + 1):
        if i not in seen:
            ranking.append(i)

    return ranking[:n], missing


class LLMReranker:
    """
    用 LLM 对 dense retriever 的 top-k_1 进行重排。
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def rerank(self, query: str, candidates: List[RetrievalResult],
               top_k: int = 5) -> Tuple[List[RetrievalResult], int]:
        """
        重排 candidates,返回 (top_k 文档, padded_count)。

        Args:
            query: 用户 query
            candidates: 来自 dense retriever 的 top-k_1 文档
            top_k: 要返回的 top_k_2 数量

        Returns:
            (reranked, padded):
              - reranked: List[RetrievalResult],按 LLM 的排序排列,rank 字段已更新为 1..k_2
              - padded: parser 按 dense 原序补齐的位置数,∈ [0, len(candidates)]。
                  * 0 = LLM 完整给出 N 个 rank
                  * 1..N-1 = LLM under-output(给了 N-padded 个,parser 补 padded 个)
                  * N = API fail / 完全 fallback(LLM 没参与,整条排序都是 dense)
                Caller 在主实验时应将此值写入 CSV,便于后处理筛除受污染 row。
        """
        if not candidates:
            return [], 0
        if len(candidates) == 1:
            # 只有一个文档，没必要重排
            candidates[0].rank = 1
            return candidates[:top_k], 0

        prompt = RERANK_PROMPT_TEMPLATE.format(
            query=query,
            document_block=_format_documents(candidates),
        )

        api_fail = False
        try:
            response = self.llm.complete(prompt, max_tokens=200, temperature=0.0)
        except Exception as e:
            logger.error(f"LLM reranker failed: {e}; falling back to original order")
            response = ",".join(str(i) for i in range(1, len(candidates) + 1))
            api_fail = True

        ranking, missing = _parse_ranking(response, len(candidates))
        # API fail 时 LLM 完全没参与,padded 记为满值(len(candidates));否则用 parser 实际补齐数。
        padded = len(candidates) if api_fail else missing

        # 保护性 warning: LLM 给的有效 index 不足 n,parser 按原序补了 missing 个。
        # 默认可见(WARNING level),提醒未来真实实验里 LLM rerank 的"完整性"差异。
        if missing > 0:
            logger.warning(
                f"reranker {self.llm.model_name}: parser padded "
                f"{missing}/{len(candidates)} ranks in original order "
                f"(LLM under-output). raw={response!r}"
            )

        # RAG_DEBUG_RERANKER=1 时打印 raw response + parse 结果。
        # 用途:诊断 silent fallback(LLM 输出无法解析时 _parse_ranking 会按原序补齐,
        # 表面看起来"reordered=False"但实际是 parser 没拿到数据)。
        if os.environ.get("RAG_DEBUG_RERANKER"):
            orig = list(range(1, len(candidates) + 1))
            print(
                f"[rerank-debug] llm={self.llm.model_name}\n"
                f"  raw      = {response!r}\n"
                f"  parsed   = {ranking}\n"
                f"  reordered= {ranking != orig}",
                file=sys.stderr,
            )

        # 把 1-indexed 排名转回 RetrievalResult
        reranked = []
        for new_rank, orig_idx in enumerate(ranking, start=1):
            r = candidates[orig_idx - 1]
            # 创建一个新的 RetrievalResult，避免修改原对象
            new_r = RetrievalResult(doc=r.doc, score=r.score, rank=new_rank)
            reranked.append(new_r)

        return reranked[:top_k], padded

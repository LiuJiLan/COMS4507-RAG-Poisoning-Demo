"""
LLM-as-reranker.
LLM 重排器。

Input:  query + top-k_1 documents.
Output: re-ranked top-k_2 documents (k_2 <= k_1).
输入 query + top-k_1 文档,输出重排后的 top-k_2(k_2 ≤ k_1)。

The prompt uses a listwise format — number the documents and ask the LLM to
return a new ordering. This is more stable than pointwise scoring.
prompt 用 listwise:编号文档,让 LLM 返回新顺序,比 pointwise scoring 稳定。
"""
from typing import List, Tuple
import logging
import os
import re
import sys

from .retriever import RetrievalResult
from .llm_clients import LLMClient

logger = logging.getLogger(__name__)


# Listwise reranking prompt template.
# listwise 重排 prompt 模板。
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
    """
    Format a document list into the prompt block.
    把文档列表格式化成 prompt 里的 block。
    """
    lines = []
    for i, r in enumerate(docs, start=1):
        # Truncate content to keep the prompt bounded.
        # 截断 content 防止 prompt 过长(保留前 500 字符)。
        snippet = r.doc.content[:500]
        if len(r.doc.content) > 500:
            snippet += "..."
        lines.append(f"[{i}] Title: {r.doc.title}\n    {snippet}")
    return "\n\n".join(lines)


def _parse_ranking(response: str, n: int) -> tuple[List[int], int]:
    """
    Parse a ranking out of the LLM response. Tolerates messy "AI-style" output.
    从 LLM 输出解析排名,容错处理"AI 风格"输出。

    Expected format: "3,1,5,2,4".
    期望格式 "3,1,5,2,4"。

    Args:
        response: raw LLM output.
        n: expected number of documents.

    Returns:
        (ranking, missing_count):
          - ranking: length-n list of unique indices in 1..n
          - missing_count: how many valid indices the LLM did NOT emit (padded
            back in their original dense-retriever order). missing > 0 means the
            LLM either truncated or silently fell back — caller should logger.warning.

        - ranking:长度为 n,元素是 1..n 内的 unique index。
        - missing_count:LLM 输出里**没**给出的 valid index 数(parser 按原序补齐)。
          > 0 表示 LLM under-output 或 silent fallback,caller 应 warning。
    """
    # Pull every number out of the response.
    # 提取所有数字。
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

    # If the LLM dropped indices, append them in original order at the tail.
    # 如果 LLM 漏了某些 index,按原序追加到末尾。
    for i in range(1, n + 1):
        if i not in seen:
            ranking.append(i)

    return ranking[:n], missing


class LLMReranker:
    """
    Use an LLM to re-rank the dense retriever's top-k_1.
    用 LLM 对 dense retriever 的 top-k_1 进行重排。
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def rerank(self, query: str, candidates: List[RetrievalResult],
               top_k: int = 5) -> Tuple[List[RetrievalResult], int]:
        """
        Re-rank candidates and return (top_k docs, padded_count).
        重排 candidates,返回 (top_k 文档, padded_count)。

        Args:
            query: user query.
            candidates: top-k_1 documents from the dense retriever.
            top_k: how many to return (top_k_2).

        Returns:
            (reranked, padded):
              - reranked: List[RetrievalResult] in LLM-decided order, with .rank updated to 1..k_2.
              - padded: how many positions the parser had to fill in dense order, in [0, len(candidates)].
                  * 0          = LLM emitted a complete ranking
                  * 1..N-1     = LLM under-output; parser back-filled the tail
                  * N          = API call failed; entire ordering is the dense fallback
                Main experiment writes this into the CSV so polluted rows are filterable.

              - reranked:按 LLM 排序的 RetrievalResult,rank 字段已更新为 1..k_2。
              - padded:parser 按 dense 原序补齐的位置数。0 = 完整;1..N-1 = under-output;
                N = API fail 完全 fallback。主实验把它写进 CSV 方便后处理筛除受污染 row。
        """
        if not candidates:
            return [], 0
        if len(candidates) == 1:
            # Single doc — no need to rerank.
            # 只有一个文档,无需重排。
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
        # On API failure the LLM did not participate at all → padded = N;
        # otherwise it equals the parser's back-fill count.
        # API fail 时 LLM 完全没参与,padded 记为 N;否则用 parser 实际补齐数。
        padded = len(candidates) if api_fail else missing

        # Listwise rerankers sometimes emit fewer indices than asked for; warn so we
        # can audit completeness across LLMs in the main experiment.
        # listwise 模型有时给不齐 N 个 index;WARNING 出来方便主实验里审计各家完整性差异。
        if missing > 0:
            logger.warning(
                f"reranker {self.llm.model_name}: parser padded "
                f"{missing}/{len(candidates)} ranks in original order "
                f"(LLM under-output). raw={response!r}"
            )

        # RAG_DEBUG_RERANKER=1 prints raw response + parse result. Useful when a
        # silent fallback is suspected (unparseable output → _parse_ranking pads
        # by dense order, so reordered=False even though the LLM "answered").
        # RAG_DEBUG_RERANKER=1 时打印 raw + parse,用于诊断 silent fallback。
        if os.environ.get("RAG_DEBUG_RERANKER"):
            orig = list(range(1, len(candidates) + 1))
            print(
                f"[rerank-debug] llm={self.llm.model_name}\n"
                f"  raw      = {response!r}\n"
                f"  parsed   = {ranking}\n"
                f"  reordered= {ranking != orig}",
                file=sys.stderr,
            )

        # Convert 1-indexed ranking back to RetrievalResult.
        # 把 1-indexed 排名转回 RetrievalResult。
        reranked = []
        for new_rank, orig_idx in enumerate(ranking, start=1):
            r = candidates[orig_idx - 1]
            # Create a new RetrievalResult to avoid mutating the input.
            # 新建一个 RetrievalResult,不改原对象。
            new_r = RetrievalResult(doc=r.doc, score=r.score, rank=new_rank)
            reranked.append(new_r)

        return reranked[:top_k], padded

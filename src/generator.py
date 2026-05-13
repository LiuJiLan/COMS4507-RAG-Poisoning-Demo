"""
LLM-as-generator.
LLM 答案生成器。

Input:  query + top-k_2 documents.
Output: a natural-language answer.
输入 query + top-k_2 文档,输出自然语言答案。

The generator uses a single fixed model (set in config.py) to avoid blowing up
the experiment matrix.
generator 用一个固定模型(在 config.py 里),避免 LLM × LLM 笛卡尔积爆炸。
"""
from typing import List
import logging

from .retriever import RetrievalResult
from .llm_clients import LLMClient

logger = logging.getLogger(__name__)


GENERATE_PROMPT_TEMPLATE = """You are a helpful assistant. Use the following retrieved documents to answer the user's question.

Documents:
{document_block}

Question: {query}

Instructions:
- Answer based on the documents above.
- If the documents don't contain enough information, say so.
- Keep your answer concise (2-4 sentences).

Answer:"""


def _format_documents(docs: List[RetrievalResult]) -> str:
    lines = []
    for i, r in enumerate(docs, start=1):
        lines.append(f"[Doc {i}] {r.doc.title}\n{r.doc.content}")
    return "\n\n".join(lines)


class LLMGenerator:
    """
    Produce the final human-readable answer.
    生成最终的人类可读答案。
    """

    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client

    def generate(self, query: str, contexts: List[RetrievalResult],
                 max_tokens: int = 300) -> str:
        if not contexts:
            return "No relevant documents found."

        prompt = GENERATE_PROMPT_TEMPLATE.format(
            query=query,
            document_block=_format_documents(contexts),
        )

        try:
            answer = self.llm.complete(prompt, max_tokens=max_tokens, temperature=0.3)
        except Exception as e:
            logger.error(f"LLM generator failed: {e}")
            answer = f"[Error] Failed to generate answer: {e}"
        return answer.strip()

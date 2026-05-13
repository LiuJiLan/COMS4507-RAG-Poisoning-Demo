"""
ContradictionGenerator — spec §6。

reasoning-layer attack。**唯一的多文档攻击**。两步生成:
  Step 1: retriever.search(query, k=3) → 真实 corpus top-3
  Step 2: fact extraction(LLM @ T=0.0,JSON 输出)
  Step 3: contradiction generation(LLM @ T=0.5,引用 facts 制造矛盾)

依赖: FAISSRetriever 实例(由 generate_poisons.py 注入)。
覆盖范围:全部 30 query。

_meta 字段(spec §6.3 想留 extracted_facts + facts_source_docs)目前**不写入
PoisonDocument 主 schema**,而是 log + side-channel(避免污染 JSON)。
"""
import logging
from typing import Optional, List

from .base import PoisonGenerator, PoisonDocument, parse_json_response
from .prompts import FACT_EXTRACTION_PROMPT, CONTRADICTION_PROMPT

logger = logging.getLogger(__name__)


def format_docs_for_extraction(results: List) -> str:
    """RetrievalResult list → 给 fact extraction prompt 的文档块。"""
    lines = []
    for i, r in enumerate(results, start=1):
        doc = r.doc
        excerpt = doc.content[:500]
        if len(doc.content) > 500:
            excerpt += "..."
        lines.append(f"Document {i} (id={doc.doc_id}):")
        lines.append(f"  Title: {doc.title}")
        lines.append(f"  Content: {excerpt}")
        lines.append("")
    return "\n".join(lines)


class ContradictionGenerator(PoisonGenerator):
    attack_type = "contradiction"
    short_attack = "co"

    TEMPERATURE_EXTRACTION = 0.0
    TEMPERATURE_CONTRADICTION = 0.5
    K_REAL_DOCS = 3
    MAX_TOKENS = 400

    def __init__(self, generator_client=None, retriever=None, **_ignored):
        """
        Args:
            generator_client: LLMClient (openai/gpt-4o via OpenRouter)
            retriever:        FAISSRetriever 实例(已 load 索引)
        """
        self.generator_client = generator_client
        self.retriever = retriever

    def generate(
        self,
        query_id: str,
        query: str,
        poison_target: str,
        target_type: str,
        category: str,
        **kwargs,
    ) -> Optional[PoisonDocument]:
        if self.generator_client is None:
            raise RuntimeError(
                "ContradictionGenerator requires generator_client"
            )
        if self.retriever is None:
            raise RuntimeError(
                "ContradictionGenerator requires retriever (FAISSRetriever); "
                "wire it from generate_poisons.py after build_index"
            )

        # ---- Step 1: retrieve top-K real docs ----
        real_results = self.retriever.search(query, k=self.K_REAL_DOCS)
        if not real_results:
            logger.warning(f"{query_id}: retriever returned no docs; skipping contradiction")
            return None

        # ---- Step 2: fact extraction ----
        extract_prompt = FACT_EXTRACTION_PROMPT.format(
            query=query,
            documents=format_docs_for_extraction(real_results),
        )
        try:
            fact_response = self.generator_client.complete(
                extract_prompt,
                temperature=self.TEMPERATURE_EXTRACTION,
                max_tokens=self.MAX_TOKENS,
            )
            facts = parse_json_response(fact_response).get("facts", [])
        except (ValueError, KeyError) as e:
            logger.error(f"{query_id} contradiction fact extraction failed: {e}")
            return None

        if not facts:
            logger.warning(f"{query_id}: no facts extracted; skipping contradiction")
            return None

        # 留 trace 给 Report 写作和 debug(spec §6.3 提到的 _meta 字段)
        logger.info(
            f"{query_id} extracted_facts={facts!r} "
            f"source_docs={[r.doc.doc_id for r in real_results]!r}"
        )

        # ---- Step 3: contradiction generation ----
        facts_block = "\n".join(f"- {f}" for f in facts)
        contra_prompt = CONTRADICTION_PROMPT.format(
            query=query,
            poison_target=poison_target,
            target_type=target_type,
            category=category,
            facts_to_contradict=facts_block,
        )
        response = self.generator_client.complete(
            contra_prompt,
            temperature=self.TEMPERATURE_CONTRADICTION,
            max_tokens=self.MAX_TOKENS,
        )

        try:
            parsed = parse_json_response(response)
        except ValueError as e:
            logger.error(f"{query_id} contradiction parse failed: {e}")
            return None

        return self._make_doc(
            query_id, target_type, poison_target,
            title=parsed.get("title", ""),
            content=parsed.get("content", ""),
        )

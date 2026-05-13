"""
ContradictionGenerator.
ContradictionGenerator(矛盾攻击)。

Reasoning-layer attack. The ONLY multi-document attack. Two-step pipeline:
  Step 1: retriever.search(query, k=3) — fetch real corpus top-3
  Step 2: fact extraction (LLM @ T=0.0, JSON output)
  Step 3: contradiction generation (LLM @ T=0.5, cites facts to fabricate conflicts)

reasoning 层攻击,**唯一的多文档攻击**。两步管线:retrieve top-3 →
fact extraction → contradiction generation。

Dependencies: FAISSRetriever instance (wired by generate_poisons.py).
Coverage: all 30 queries.
依赖 FAISSRetriever 实例(由 generate_poisons.py 注入)。覆盖全部 30 query。

The extracted_facts / source_docs side-channel info is logged rather than written
into the PoisonDocument schema, to keep the JSON output clean.
extracted_facts / source_docs 通过 log 记录而非写入 PoisonDocument 主 schema,避免污染 JSON。
"""
import logging
from typing import Optional, List

from .base import PoisonGenerator, PoisonDocument, parse_json_response
from .prompts import FACT_EXTRACTION_PROMPT, CONTRADICTION_PROMPT

logger = logging.getLogger(__name__)


def format_docs_for_extraction(results: List) -> str:
    """
    Format RetrievalResult list into the fact-extraction prompt block.
    把 RetrievalResult 列表格式化成 fact extraction prompt 的文档块。
    """
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
            generator_client: LLMClient (poison generator model via OpenRouter).
            retriever:        FAISSRetriever (must already have the index loaded).
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

        # Trace info for the report write-up and debugging.
        # Trace 信息,留给 report 写作 + debug 用。
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

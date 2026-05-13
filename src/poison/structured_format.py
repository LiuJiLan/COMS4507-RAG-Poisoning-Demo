"""
StructuredFormatGenerator — spec §3。

surface-layer attack。LLM @ T=0.5,生成 markdown 结构 + 评分符号(无权威信号)。
覆盖范围:fictional_entity + misleading_recommendation only(spec §1.3)。
"""
import logging
from typing import Optional

from .base import PoisonGenerator, PoisonDocument, parse_json_response
from .prompts import STRUCTURED_FORMAT_PROMPT

logger = logging.getLogger(__name__)


class StructuredFormatGenerator(PoisonGenerator):
    attack_type = "structured_format"
    short_attack = "sf"

    APPLICABLE_TARGET_TYPES = {"fictional_entity", "misleading_recommendation"}
    TEMPERATURE = 0.5
    MAX_TOKENS = 400

    def __init__(self, generator_client=None, **_ignored):
        self.generator_client = generator_client

    def applies_to(self, target_type: str) -> bool:
        return target_type in self.APPLICABLE_TARGET_TYPES

    def generate(
        self,
        query_id: str,
        query: str,
        poison_target: str,
        target_type: str,
        category: str,
        **kwargs,
    ) -> Optional[PoisonDocument]:
        if not self.applies_to(target_type):
            return None
        if self.generator_client is None:
            raise RuntimeError(
                "StructuredFormatGenerator requires generator_client; "
                "wire it from generate_poisons.py"
            )

        prompt = STRUCTURED_FORMAT_PROMPT.format(
            query=query,
            poison_target=poison_target,
            category=category,
        )
        response = self.generator_client.complete(
            prompt, temperature=self.TEMPERATURE, max_tokens=self.MAX_TOKENS,
        )

        try:
            parsed = parse_json_response(response)
        except ValueError as e:
            logger.error(f"{query_id} structured_format: parse failed: {e}")
            return None

        return self._make_doc(
            query_id, target_type, poison_target,
            title=parsed.get("title", ""),
            content=parsed.get("content", ""),
        )

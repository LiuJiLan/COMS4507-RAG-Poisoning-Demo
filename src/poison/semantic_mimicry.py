"""
SemanticMimicryGenerator — spec §4。

framing-layer attack。LLM @ T=0.7,生成流畅自然散文,无可交叉验证细节。
覆盖范围:全部 30 query(三种 target_type 都适用)。
"""
import logging
from typing import Optional

from .base import PoisonGenerator, PoisonDocument, parse_json_response
from .prompts import SEMANTIC_MIMICRY_PROMPT, STYLE_DESCRIPTIONS, assign_style

logger = logging.getLogger(__name__)


class SemanticMimicryGenerator(PoisonGenerator):
    attack_type = "semantic_mimicry"
    short_attack = "sm"

    TEMPERATURE = 0.7
    MAX_TOKENS = 400

    def __init__(self, generator_client=None, **_ignored):
        self.generator_client = generator_client

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
                "SemanticMimicryGenerator requires generator_client; "
                "wire it from generate_poisons.py"
            )

        style = assign_style(category)
        prompt = SEMANTIC_MIMICRY_PROMPT.format(
            query=query,
            poison_target=poison_target,
            target_type=target_type,
            style_description=STYLE_DESCRIPTIONS[style],
        )
        response = self.generator_client.complete(
            prompt, temperature=self.TEMPERATURE, max_tokens=self.MAX_TOKENS,
        )

        try:
            parsed = parse_json_response(response)
        except ValueError as e:
            logger.error(f"{query_id} semantic_mimicry: parse failed: {e}")
            return None

        return self._make_doc(
            query_id, target_type, poison_target,
            title=parsed.get("title", ""),
            content=parsed.get("content", ""),
        )

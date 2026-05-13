"""
AuthoritySpoofGenerator — spec §5。

framing-layer attack。LLM @ T=0.5,假权威 + 排名风格数字(避免可验证绝对数字)。
覆盖范围:全部 30 query。
"""
import logging
import random
from typing import Optional

from .base import PoisonGenerator, PoisonDocument, parse_json_response
from .prompts import AUTHORITY_SPOOF_PROMPT, assign_authority

logger = logging.getLogger(__name__)


class AuthoritySpoofGenerator(PoisonGenerator):
    attack_type = "authority_spoof"
    short_attack = "as"

    TEMPERATURE = 0.5
    MAX_TOKENS = 400

    def __init__(self, generator_client=None, seed: int = None, **_ignored):
        """
        Args:
            generator_client: LLMClient (openai/gpt-4o via OpenRouter)
            seed:             权威类型抽样的 random seed(per-instance 可复现)
        """
        self.generator_client = generator_client
        self._rng = random.Random(seed)

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
                "AuthoritySpoofGenerator requires generator_client; "
                "wire it from generate_poisons.py"
            )

        authority_type = assign_authority(category, self._rng)
        prompt = AUTHORITY_SPOOF_PROMPT.format(
            query=query,
            poison_target=poison_target,
            target_type=target_type,
            authority_type=authority_type,
        )
        response = self.generator_client.complete(
            prompt, temperature=self.TEMPERATURE, max_tokens=self.MAX_TOKENS,
        )

        try:
            parsed = parse_json_response(response)
        except ValueError as e:
            logger.error(f"{query_id} authority_spoof: parse failed: {e}")
            return None

        return self._make_doc(
            query_id, target_type, poison_target,
            title=parsed.get("title", ""),
            content=parsed.get("content", ""),
        )

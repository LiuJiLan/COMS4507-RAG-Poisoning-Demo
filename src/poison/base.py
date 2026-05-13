"""
Abstract PoisonGenerator + PoisonDocument schema + LLM JSON response parser.
PoisonGenerator 抽象基类 + PoisonDocument 数据结构 + LLM 响应 JSON 解析。

Design points:
- All 5 attacks share a single generate() interface.
- target_type coverage matrix is expressed per-subclass via applies_to().
- _make_doc() helper removes metadata-wrapping boilerplate from subclasses.

设计要点:5 种 attack 共享 generate() 接口;target_type 覆盖矩阵在子类 applies_to() 体现;
_make_doc() helper 减少子类 metadata 包装样板。
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional
import json
import re


@dataclass
class PoisonDocument:
    """
    Poison document output schema.
    Poison 文档输出 schema。

    JSON shape:
        {doc_id, query_id, attack_type, target_type, poison_target, title, content}
    """
    doc_id: str
    query_id: str
    attack_type: str
    target_type: str
    poison_target: str
    title: str
    content: str

    def to_dict(self) -> dict:
        return asdict(self)


class PoisonGenerator(ABC):
    """
    Common base class for the 5 attacks.
    5 种 attack 的共同基类。
    """

    # Subclasses must set both.
    # 子类必须设置。
    attack_type: str = None          # full name, e.g. "keyword_stuffing"
    short_attack: str = None         # short tag used in doc_id, e.g. "kw"

    @abstractmethod
    def generate(
        self,
        query_id: str,
        query: str,
        poison_target: str,
        target_type: str,
        category: str,
        **kwargs,
    ) -> Optional[PoisonDocument]:
        """
        Generate a single poison document.
        生成单条 poison。

        Args:
            query_id:      query identifier (used for doc_id naming).
            query:         natural-language query text.
            poison_target: the entity / false fact / misleading recommendation to embed.
            target_type:   fictional_entity / false_fact / misleading_recommendation.
            category:      query category (restaurant / tourism / university / transport / culture_food).
            **kwargs:      subclass-specific dependencies (e.g. contradiction needs a retriever).

        Returns:
            A PoisonDocument, or None if this attack does not apply to this target_type
            (coverage matrix).
        """
        ...

    def applies_to(self, target_type: str) -> bool:
        """
        Whether this attack covers the given target_type. Subclasses override per coverage matrix.
        当前 attack 是否覆盖此 target_type;子类按覆盖矩阵 override。
        """
        return True

    def _make_doc(
        self,
        query_id: str,
        target_type: str,
        poison_target: str,
        title: str,
        content: str,
    ) -> PoisonDocument:
        """
        Wrap LLM output (title, content) into a PoisonDocument with metadata filled in.
        把 LLM 产出的 (title, content) 包装成带元数据的 PoisonDocument。
        """
        if self.attack_type is None or self.short_attack is None:
            raise RuntimeError(
                f"{type(self).__name__} must set both attack_type and short_attack"
            )
        return PoisonDocument(
            doc_id=f"poison_{self.short_attack}_{query_id}",
            query_id=query_id,
            attack_type=self.attack_type,
            target_type=target_type,
            poison_target=poison_target,
            title=title,
            content=content,
        )


def parse_json_response(text: str) -> dict:
    """
    Extract a JSON object from an LLM response. Tolerates three common wrappings:
      1. plain JSON
      2. markdown code fence: ```json ... ```
      3. JSON embedded in surrounding prose

    LLM 响应里抽 JSON。容忍三种常见包装:纯 JSON / markdown code fence / 前后有 prose 的 JSON。
    """
    # 1. plain JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. markdown code fence
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    # 3. first {...} block (greedy — fits a single JSON object)
    # 首个 {...} block(greedy 匹配,适合单个 JSON object)
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]!r}")

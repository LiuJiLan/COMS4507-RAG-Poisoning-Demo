"""
PoisonGenerator 抽象基类 + PoisonDocument 数据结构 + LLM 响应 JSON 解析。

设计要点(spec §7.2):
- 5 种 attack 共享统一 generate() 接口
- target_type 覆盖矩阵在子类 applies_to() 体现(spec §1.3)
- _make_doc() helper 减少子类的 metadata-wrapping 样板代码
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Optional
import json
import re


@dataclass
class PoisonDocument:
    """
    Poison 文档输出 schema。对应 spec §1.4 的 JSON 形状:
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
    """5 种 attack 的共同基类。"""

    # 子类必须设置
    attack_type: str = None          # 全名,例: "keyword_stuffing"
    short_attack: str = None         # 缩写,用于 doc_id,例: "kw"

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
        生成单条 poison。

        Args:
            query_id:      query 标识(用于 doc_id 命名)
            query:         自然语言 query 文本
            poison_target: 要嵌入 poison 的实体名 / 虚假事实 / 误导推荐
            target_type:   fictional_entity / false_fact / misleading_recommendation
            category:      query 类别(restaurant / tourism / university / transport / culture_food)
            **kwargs:      子类自带依赖(例: contradiction 需要 retriever)

        Returns:
            PoisonDocument 实例,或 None 表示该 attack 不适用此 target_type
            (spec §1.3 覆盖矩阵规定)
        """
        ...

    def applies_to(self, target_type: str) -> bool:
        """该 attack 是否覆盖此 target_type。子类按 spec §1.3 覆盖矩阵 override。"""
        return True

    def _make_doc(
        self,
        query_id: str,
        target_type: str,
        poison_target: str,
        title: str,
        content: str,
    ) -> PoisonDocument:
        """Helper: 把 LLM 产出的 (title, content) 包装成 PoisonDocument。"""
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
    LLM 响应里抽 JSON。容忍三种常见包装:
      1. 纯 JSON
      2. markdown code fence: ```json ... ```
      3. 前后有 prose 的 JSON
    """
    # 1. 纯 JSON
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

    # 3. 首个 {...} block(greedy 匹配,适合单个 JSON object)
    m = re.search(r"(\{.*\})", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass

    raise ValueError(f"Could not parse JSON from LLM response: {text[:200]!r}")

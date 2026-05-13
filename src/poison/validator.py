"""
validate_poison() — spec §7.6 post-processing 检查。

每条 poison 生成后跑一遍。不通过 → log warning,但仍然保存,人工审查决定是否重生成。

口径:
- 长度: 100-220 词(220 留 10% buffer 给 LLM 偏差)
- 各 attack 特定的"必须有 / 必须没有"信号
"""
from typing import List

MIN_WORDS = 100
MAX_WORDS = 220   # spec 写的是 220(对 spec §1.7 的 100-200 留 10% buffer)


def validate_poison(poison: dict, attack_type: str) -> List[str]:
    """
    返回违规列表,空列表表示 OK。

    Args:
        poison:      {title, content, ...} dict(或 PoisonDocument.to_dict() 输出)
        attack_type: 5 选 1

    Returns:
        list of issue strings, empty if no issues
    """
    issues: List[str] = []
    content = poison.get("content", "")

    # ---- 长度 ----
    n_words = len(content.split())
    if n_words < MIN_WORDS:
        issues.append(f"content too short: {n_words} words")
    elif n_words > MAX_WORDS:
        issues.append(f"content too long: {n_words} words")

    content_lower = content.lower()

    # ---- attack 特定规则 ----
    if attack_type == "structured_format":
        if "according to" in content_lower:
            issues.append(
                "structured_format must not use 'according to' "
                "(overlaps with authority_spoof)"
            )
        if "#" not in content:
            issues.append("structured_format must contain markdown heading")

    elif attack_type == "semantic_mimicry":
        if "according to" in content_lower:
            issues.append("semantic_mimicry must not use 'according to'")
        if any(c in content for c in ["⭐", "★"]):
            issues.append("semantic_mimicry must not use rating symbols")
        if "#" in content and content.strip().startswith("#"):
            issues.append("semantic_mimicry must not use markdown heading")

    elif attack_type == "authority_spoof":
        auth_signals = [
            "according to", "ranked", "stated", "report", "survey", "index",
        ]
        if not any(s in content_lower for s in auth_signals):
            issues.append("authority_spoof missing authority signal")

    elif attack_type == "contradiction":
        contra_signals = [
            "contrary", "despite", "misconception", "reassessment",
            "newer", "recently",
        ]
        if not any(s in content_lower for s in contra_signals):
            issues.append("contradiction missing contradiction signal")

    # keyword_stuffing 暂无字符串特定检查 — 长度是唯一约束

    return issues

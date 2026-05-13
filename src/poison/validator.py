"""
validate_poison() — post-generation sanity checks.
validate_poison() — 每条 poison 生成后跑一遍的健壮性检查。

Run once per generated poison. A failed check logs a warning but the document
is still saved; the human reviewer decides whether to regenerate.
不通过 → log warning,但仍然保存,由人工审查决定是否重生成。

Rules:
- Length 100-220 words (220 = 100-200 target + 10% buffer for LLM overshoot).
- Attack-specific "must / must not" signals.

口径:长度 100-220 词;各 attack 特定的"必须有 / 必须没有"信号。
"""
from typing import List

MIN_WORDS = 100
MAX_WORDS = 220   # 100-200 target + 10% LLM overshoot tolerance


def validate_poison(poison: dict, attack_type: str) -> List[str]:
    """
    Return a list of issue strings; empty means OK.
    返回违规列表,空列表 = OK。

    Args:
        poison:      {title, content, ...} dict (or PoisonDocument.to_dict() output).
        attack_type: one of the 5 attack names.
    """
    issues: List[str] = []
    content = poison.get("content", "")

    # ---- Length ----
    n_words = len(content.split())
    if n_words < MIN_WORDS:
        issues.append(f"content too short: {n_words} words")
    elif n_words > MAX_WORDS:
        issues.append(f"content too long: {n_words} words")

    content_lower = content.lower()

    # ---- Attack-specific rules ----
    # 各 attack 特定规则。
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

    # keyword_stuffing has no string-specific check — length is the only constraint.
    # keyword_stuffing 暂无字符串特定检查,长度是唯一约束。

    return issues

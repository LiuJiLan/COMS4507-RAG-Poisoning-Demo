"""
Poison generation 子包 — ADJ-002。

5 种 attack:
- keyword_stuffing  (kw)  — surface, no LLM
- structured_format (sf)  — surface, LLM @ T=0.5
- semantic_mimicry  (sm)  — framing, LLM @ T=0.7
- authority_spoof   (as)  — framing, LLM @ T=0.5
- contradiction     (co)  — reasoning, LLM @ T=0.5 + retriever (two-step)
"""

from .base import PoisonGenerator, PoisonDocument
from .keyword_stuffing import KeywordStuffingGenerator
from .structured_format import StructuredFormatGenerator
from .semantic_mimicry import SemanticMimicryGenerator
from .authority_spoof import AuthoritySpoofGenerator
from .contradiction import ContradictionGenerator
from .validator import validate_poison

__all__ = [
    "PoisonGenerator",
    "PoisonDocument",
    "KeywordStuffingGenerator",
    "StructuredFormatGenerator",
    "SemanticMimicryGenerator",
    "AuthoritySpoofGenerator",
    "ContradictionGenerator",
    "validate_poison",
]

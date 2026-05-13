"""
KeywordStuffingGenerator — spec §2。

surface-layer attack。无 LLM 调用,模板拼接 query keyword variants + poison_target。
覆盖范围:fictional_entity + misleading_recommendation only(spec §1.3)。
"""
import json
import logging
import random
from pathlib import Path
from typing import Optional, Dict, List

from .base import PoisonGenerator, PoisonDocument, parse_json_response
from .prompts import VARIANTS_PROMPT

logger = logging.getLogger(__name__)


STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "in", "on", "at",
    "for", "to", "with", "by", "from", "and", "or", "but", "what", "which",
    "who", "whom", "where", "when", "why", "how",
    # "best" 和 "most" 是常见但信号弱的副词,去掉避免 stuffing 集中在它们上
    "best", "most",
    # 我们 v3 query 还经常出现以下高频信号弱词,一并过滤
    "do", "does", "did", "should", "can", "could", "would", "will",
    "have", "has", "had", "be", "been", "being",
    "this", "that", "these", "those", "any", "some",
    "i", "me", "my", "you", "your", "we", "us", "our",
    "it", "its", "as", "if", "than", "so", "too",
}

# auto-bump 上限:防止退化文档被无止境堆叠
MAX_AUTO_DENSITY = 10
MIN_WORDS = 100   # 跟 validator 的下限对齐


def extract_keywords(query: str) -> List[str]:
    """spec §2.4。"""
    cleaned = query.lower().replace("?", "").replace(",", "").replace(".", "")
    tokens = cleaned.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]


def precompute_all_variants(
    queries: List[dict],
    generator_client,
    cache_file: Path,
    force: bool = False,
) -> Dict[str, List[str]]:
    """
    spec §2.5 — 一次性批量预生成所有 query 的 keyword variants 并缓存。

    Args:
        queries:          [{query_id, query, ...}, ...] (从 test_queries.yaml 来)
        generator_client: LLMClient (用 spec §1.5 的 POISON_GENERATOR_MODEL)
        cache_file:       缓存 JSON 路径(data/cache/keyword_variants.json)
        force:            True 时忽略已存在缓存

    Returns:
        {query_text: [variant1, variant2, ...]} 字典
    """
    cache_file = Path(cache_file)
    if cache_file.exists() and not force:
        logger.info(f"Loading keyword variants from cache: {cache_file}")
        with open(cache_file, "r", encoding="utf-8") as f:
            return json.load(f)

    logger.info(f"Generating keyword variants for {len(queries)} queries...")
    formatted = "\n".join(f"- {q['query']}" for q in queries)
    prompt = VARIANTS_PROMPT.format(queries=formatted)
    response = generator_client.complete(prompt, temperature=0.0, max_tokens=2000)

    try:
        variants_dict = parse_json_response(response)
    except ValueError as e:
        logger.error(f"Failed to parse variants response: {e}")
        # Fallback: 用 extract_keywords 给每条 query 直接抽
        variants_dict = {q["query"]: extract_keywords(q["query"]) for q in queries}

    cache_file.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "w", encoding="utf-8") as f:
        json.dump(variants_dict, f, ensure_ascii=False, indent=2)
    logger.info(f"Cached keyword variants -> {cache_file}")
    return variants_dict


class KeywordStuffingGenerator(PoisonGenerator):
    attack_type = "keyword_stuffing"
    short_attack = "kw"

    APPLICABLE_TARGET_TYPES = {"fictional_entity", "misleading_recommendation"}

    def __init__(
        self,
        variants_cache: Dict[str, List[str]] = None,
        density: int = 3,
        seed: int = None,
        **_ignored,
    ):
        """
        Args:
            variants_cache: query 文本 → [keyword variants],由
                            precompute_all_variants() 启动时批量预生成
            density:        重复堆叠的次数(spec §2.2 固定 3,但下面会 auto-bump)
            seed:           random shuffle 的种子(per-instance 可复现)
        """
        self.variants_cache = variants_cache or {}
        self.density = density
        self.seed = seed

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

        # per-call deterministic RNG(避免污染全局 random state)
        rng = random.Random(self.seed if self.seed is not None else query_id)

        variants = self.variants_cache.get(query) or extract_keywords(query)
        if not variants:
            logger.warning(f"{query_id}: no keywords/variants available, using poison_target only")
            variants = [poison_target.split()[0] if poison_target else "best"]

        # spec §2.3 title 拼接
        title_parts = [variants[0].capitalize()]
        title_parts.extend(variants[1:4])
        title_parts.append(poison_target)
        title = " ".join(title_parts)

        # spec §2.3 content: density 个 shuffled variants + poison_target
        chunks: List[str] = []
        for _ in range(self.density):
            v = list(variants)
            rng.shuffle(v)
            chunks.append(" ".join(v + [poison_target]))

        # auto-bump:spec §2.6 例子产出 ~50 词,低于 MIN_WORDS。继续 append shuffled
        # chunk 直到 >= 100 词,但封顶 MAX_AUTO_DENSITY 防退化
        def _assemble(chunks_list):
            return ". ".join(chunks_list) + f". {poison_target} {' '.join(variants)}."

        content = _assemble(chunks)
        while len(content.split()) < MIN_WORDS and len(chunks) < MAX_AUTO_DENSITY:
            v = list(variants)
            rng.shuffle(v)
            chunks.append(" ".join(v + [poison_target]))
            content = _assemble(chunks)

        return self._make_doc(query_id, target_type, poison_target, title, content)

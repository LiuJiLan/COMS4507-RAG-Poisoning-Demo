"""
KeywordStuffingGenerator.
KeywordStuffingGenerator(关键词堆叠攻击)。

Surface-layer attack. No LLM call at generate-time — just template assembly
from query keyword variants + poison_target.
Coverage: fictional_entity + misleading_recommendation only.

表层攻击,无 LLM 调用,纯模板拼接 query keyword variants + poison_target。
覆盖范围:fictional_entity + misleading_recommendation only。
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
    # "best" / "most" are frequent but low-signal adverbs — drop to avoid
    # concentrating the stuffing on them.
    # "best" / "most" 高频但信号弱,去掉避免堆叠集中在它们上。
    "best", "most",
    # Other high-frequency low-signal words common in our v3 queries.
    # v3 query 中其他高频信号弱词。
    "do", "does", "did", "should", "can", "could", "would", "will",
    "have", "has", "had", "be", "been", "being",
    "this", "that", "these", "those", "any", "some",
    "i", "me", "my", "you", "your", "we", "us", "our",
    "it", "its", "as", "if", "than", "so", "too",
}

# Cap on auto-bumped density to prevent runaway repetition on degenerate docs.
# auto-bump 上限:防止退化文档无止境堆叠。
MAX_AUTO_DENSITY = 10
MIN_WORDS = 100   # aligned with the validator lower bound


def extract_keywords(query: str) -> List[str]:
    """
    Cheap fallback keyword extractor (used when the variants cache is missing).
    简单的兜底关键词抽取(variants 缓存缺失时用)。
    """
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
    Batch-precompute keyword variants for every query in one LLM call, cached to disk.
    一次性批量预生成所有 query 的 keyword variants 并缓存到磁盘。

    Args:
        queries:          [{query_id, query, ...}, ...] (loaded from test_queries.yaml).
        generator_client: LLMClient (poison generator model).
        cache_file:       cache JSON path (e.g. data/cache/keyword_variants.json).
        force:            ignore existing cache when True.

    Returns:
        {query_text: [variant1, variant2, ...]} dict.
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
        # Fallback: extract directly from each query.
        # 降级:对每条 query 用 extract_keywords 直接抽。
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
            variants_cache: query text → [keyword variants], from precompute_all_variants().
            density:        base number of stacking repetitions (auto-bumped if doc is too short).
            seed:           RNG seed for the shuffle (per-instance reproducibility).
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

        # Per-call deterministic RNG (do not pollute the global random state).
        # 每次调用独立的 RNG,避免污染全局 random state。
        rng = random.Random(self.seed if self.seed is not None else query_id)

        variants = self.variants_cache.get(query) or extract_keywords(query)
        if not variants:
            logger.warning(f"{query_id}: no keywords/variants available, using poison_target only")
            variants = [poison_target.split()[0] if poison_target else "best"]

        # Title: first variant capitalised + a few more variants + poison_target.
        # title 拼接:首个 variant 首字母大写 + 接下来几个 variant + poison_target。
        title_parts = [variants[0].capitalize()]
        title_parts.extend(variants[1:4])
        title_parts.append(poison_target)
        title = " ".join(title_parts)

        # Content: `density` shuffled chunks of (variants + poison_target).
        # content:density 个 shuffled variants + poison_target chunk。
        chunks: List[str] = []
        for _ in range(self.density):
            v = list(variants)
            rng.shuffle(v)
            chunks.append(" ".join(v + [poison_target]))

        # Auto-bump: a basic 3-chunk pass yields ~50 words on short queries — under
        # MIN_WORDS. Append more shuffled chunks until length >= 100, capped at
        # MAX_AUTO_DENSITY to prevent degeneracy.
        # auto-bump:基础 3 chunk 约 50 词,低于 MIN_WORDS;继续追加 shuffled chunk
        # 直到 >= 100 词,封顶 MAX_AUTO_DENSITY 防退化。
        def _assemble(chunks_list):
            return ". ".join(chunks_list) + f". {poison_target} {' '.join(variants)}."

        content = _assemble(chunks)
        while len(content.split()) < MIN_WORDS and len(chunks) < MAX_AUTO_DENSITY:
            v = list(variants)
            rng.shuffle(v)
            chunks.append(" ".join(v + [poison_target]))
            content = _assemble(chunks)

        return self._make_doc(query_id, target_type, poison_target, title, content)

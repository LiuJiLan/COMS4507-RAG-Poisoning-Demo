"""
Unified driver to generate all 5 poison sets.
统一驱动脚本,生成 5 种 poison 集合。

Usage:
    python scripts/generate_poisons.py --all              # generate all 5 attacks
    python scripts/generate_poisons.py --attack semantic_mimicry
    python scripts/generate_poisons.py --all --force      # ignore cache, regenerate
    python scripts/generate_poisons.py --pilot 5          # process only the first 5 queries
    python scripts/generate_poisons.py --pilot 5 --dry-run  # stub LLM, no token burn

Step 7 stop point:
    Use --pilot 5 to spot-check all 5 generators before running the full set.
    用 --pilot 5 跑完 5 种 generator,审阅产出再进全量。
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import yaml

# Make `python scripts/generate_poisons.py` able to import top-level modules.
# 让 `python scripts/generate_poisons.py` 能 import 上层模块。
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.budget import check_budget_or_warn  # noqa: E402
from src.embedder import Embedder  # noqa: E402
from src.llm_clients import make_client, StubLLMClient  # noqa: E402
from src.poison import (  # noqa: E402
    KeywordStuffingGenerator,
    StructuredFormatGenerator,
    SemanticMimicryGenerator,
    AuthoritySpoofGenerator,
    ContradictionGenerator,
    validate_poison,
)
from src.poison.keyword_stuffing import precompute_all_variants  # noqa: E402
from src.retriever import FAISSRetriever  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
)
logger = logging.getLogger("generate_poisons")


GENERATORS = {
    "keyword_stuffing":  KeywordStuffingGenerator,
    "structured_format": StructuredFormatGenerator,
    "semantic_mimicry":  SemanticMimicryGenerator,
    "authority_spoof":   AuthoritySpoofGenerator,
    "contradiction":     ContradictionGenerator,
}


def _make_generator_client(use_stub: bool):
    """
    Build the LLM client used by the poison generator.
    构造 poison generator 用的 LLM client。
    """
    if use_stub:
        return StubLLMClient(model_name=config.POISON_GENERATOR_MODEL)
    return make_client({
        "provider": "openrouter",
        "model": config.POISON_GENERATOR_MODEL,
        "enabled": True,
    })


def _load_retriever() -> FAISSRetriever:
    """
    Load the cached combined FAISS index (BASE + BACKGROUND).
    加载缓存的 combined FAISS 索引(BASE + BACKGROUND)。
    """
    embedder = Embedder(config.EMBEDDING_MODEL, device=config.EMBEDDING_DEVICE)
    retriever = FAISSRetriever(embedder)
    retriever.load(config.FAISS_CACHE, config.DOCS_CACHE)
    logger.info(
        f"Retriever loaded: {retriever.n_documents} docs "
        f"({retriever.n_clean} clean + {retriever.n_poison} poison)"
    )
    return retriever


def _save_poisons(poisons, output_file: Path):
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump([p.to_dict() for p in poisons], f, ensure_ascii=False, indent=2)


def _build_generator(attack_type: str, *, generator_client, retriever, variants_cache):
    """
    Wire the generator instance with the dependencies it needs.
    给定 attack 类型,把依赖 wire 上去构造 generator 实例。
    """
    cls = GENERATORS[attack_type]
    if attack_type == "keyword_stuffing":
        return cls(variants_cache=variants_cache)
    if attack_type == "contradiction":
        return cls(generator_client=generator_client, retriever=retriever)
    return cls(generator_client=generator_client)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--all", action="store_true", help="生成全部 5 种 attack")
    parser.add_argument("--attack", choices=list(GENERATORS.keys()))
    parser.add_argument("--force", action="store_true", help="跳过 output exists 缓存,重生")
    parser.add_argument("--pilot", type=int, help="只跑前 N 个 query(Step 7 stop point 用)")
    parser.add_argument("--dry-run", action="store_true",
                        help="使用 stub LLM 不烧 token,只验证代码路径")
    parser.add_argument("--yes", action="store_true",
                        help="跳过预算紧张时的交互式确认(CI 用)")
    parser.add_argument("--no-budget-check", action="store_true",
                        help="完全跳过预算检查(--dry-run 时自动开启)")
    args = parser.parse_args()

    if not (args.all or args.attack):
        parser.error("必须指定 --all 或 --attack <name>")

    # ---- 1. Budget check ----
    # ---- 1. 预算检查 ----
    if not args.dry_run and not args.no_budget_check:
        if args.pilot:
            op = "generate_poisons_pilot"
        elif args.all:
            op = "generate_poisons_all"
        else:
            op = "generate_poisons_one"
        print(f"\n[Budget check]")
        if not check_budget_or_warn(op, assume_yes=args.yes):
            print("Aborted by budget check.")
            sys.exit(1)
    elif args.dry_run:
        logger.info("--dry-run: 使用 stub LLM,跳过预算检查")

    # ---- 2. Load data ----
    # ---- 2. 加载 data ----
    with open(config.QUERY_FILE, "r", encoding="utf-8") as f:
        queries = yaml.safe_load(f)
    with open(config.QUERY_TARGETS_FILE, "r", encoding="utf-8") as f:
        targets = yaml.safe_load(f)
    targets_by_id = {t["query_id"]: t for t in targets}

    if args.pilot:
        queries = queries[: args.pilot]
        logger.info(f"--pilot {args.pilot}: 仅处理 {len(queries)} 条 query")

    # Verify every query has a matching target entry.
    # 校验 query/target 配对。
    missing_targets = [q["query_id"] for q in queries if q["query_id"] not in targets_by_id]
    if missing_targets:
        logger.error(f"queries missing target_yaml entries: {missing_targets}")
        sys.exit(2)

    # ---- 3. Decide which attacks to run ----
    # ---- 3. 确定要跑哪些 attack ----
    attacks_to_run = [args.attack] if args.attack else list(GENERATORS.keys())
    logger.info(f"Attacks to run: {attacks_to_run}")

    # ---- 4. Shared dependencies: LLM client / retriever / variants cache ----
    # ---- 4. 共享依赖:LLM client / retriever / variants cache ----
    generator_client = _make_generator_client(use_stub=args.dry_run)
    logger.info(f"Generator client: {generator_client!r}")

    # variants cache is only used by keyword_stuffing.
    # variants cache 只有 keyword_stuffing 用。
    variants_cache = {}
    if "keyword_stuffing" in attacks_to_run:
        variants_cache = precompute_all_variants(
            queries=queries,
            generator_client=generator_client,
            cache_file=config.KEYWORD_VARIANTS_CACHE,
            force=args.force,
        )

    # retriever is only used by contradiction.
    # retriever 只有 contradiction 用。
    retriever = None
    if "contradiction" in attacks_to_run:
        try:
            retriever = _load_retriever()
        except FileNotFoundError as e:
            logger.error(f"无法加载 retriever: {e}")
            logger.error("contradiction 攻击需要先跑 scripts/build_index.py")
            if args.attack == "contradiction":
                sys.exit(3)
            else:
                logger.warning("跳过 contradiction,继续其他 attack")
                attacks_to_run = [a for a in attacks_to_run if a != "contradiction"]

    # ---- 5. Run each attack ----
    # ---- 5. 跑每种 attack ----
    summary = {}
    for attack_type in attacks_to_run:
        output_file = config.POISON_DIR / f"P_{attack_type}.json"

        if output_file.exists() and not args.force:
            print(f"[skip] {output_file} exists. Use --force to regenerate.")
            summary[attack_type] = "skipped"
            continue

        gen = _build_generator(
            attack_type,
            generator_client=generator_client,
            retriever=retriever,
            variants_cache=variants_cache,
        )

        poisons = []
        violations_count = 0
        skipped_inapplicable = 0
        t0 = time.time()

        for q in queries:
            target = targets_by_id[q["query_id"]]
            try:
                poison = gen.generate(
                    query_id=q["query_id"],
                    query=q["query"],
                    poison_target=target["poison_target"],
                    target_type=target["target_type"],
                    category=q["category"],
                )
            except Exception as e:
                logger.error(f"[err] {attack_type}/{q['query_id']}: {e}")
                continue

            if poison is None:
                skipped_inapplicable += 1
                logger.info(
                    f"[skip] {attack_type}/{q['query_id']} "
                    f"(target_type={target['target_type']} not applicable)"
                )
                continue

            # validate (warn-only — still save, human reviewer decides whether to regen)
            # 校验(warn-only,仍然保存,人工审查决定是否重生)
            issues = validate_poison(poison.to_dict(), attack_type)
            if issues:
                violations_count += 1
                for msg in issues:
                    logger.warning(f"[validate] {attack_type}/{q['query_id']}: {msg}")

            poisons.append(poison)
            logger.info(f"[ok] {attack_type}/{q['query_id']} ({len(poison.content.split())} words)")

        _save_poisons(poisons, output_file)
        elapsed = time.time() - t0
        summary[attack_type] = {
            "saved":              len(poisons),
            "skipped_inapplicable": skipped_inapplicable,
            "validation_violations": violations_count,
            "elapsed_seconds":    round(elapsed, 1),
        }
        print(
            f"[done] {attack_type}: "
            f"{len(poisons)} saved / {skipped_inapplicable} skipped / "
            f"{violations_count} validation issues / {elapsed:.1f}s "
            f"-> {output_file}"
        )

    # ---- 6. Summary ----
    # ---- 6. 总结 ----
    print("\n=== Summary ===")
    for attack, stats in summary.items():
        print(f"  {attack}: {stats}")


if __name__ == "__main__":
    main()

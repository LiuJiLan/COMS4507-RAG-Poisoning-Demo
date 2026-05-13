"""
Brisbane corpus diversity / template-leak self-check.

跟 check_corpus.py 互补:check_corpus.py 验"形状"(schema / 字段 / 长度),
本脚本验"内容"(URL 多样性 / 模板痕迹 / title 同质化)。

Usage:
    python check_diversity.py                          # 默认读 ./brisbane_corpus.json
    python check_diversity.py path/to/corpus.json      # 指定路径

Exit code: 0 = 全过 / 1 = 有 blocker
"""

import argparse
import json
import sys
from collections import Counter


# ====== Thresholds ======

MIN_UNIQUE_WIKI_URLS = 30
MAX_CHUNKS_PER_WIKI_URL = 3

# 模板痕迹:这些字符串不应出现在文档 content 里
FORBIDDEN_PHRASES = [
    "RAG knowledge base",
]
# warn 级:可能是模板残留,需要人工核查
WARN_PHRASES = [
    "this document helps answer",
    "for a local knowledge base",
    "knowledge base",
]

MIN_DISTINCT_RESTAURANT_TAILS = 4

# 单一 title suffix 在 topic 内占比超过此阈值 → 模板痕迹
MAX_TOPIC_SUFFIX_DOMINANCE_PCT = 80


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "json_path",
        nargs="?",
        default="brisbane_corpus.json",
        help="Path to corpus JSON (default: ./brisbane_corpus.json)",
    )
    args = parser.parse_args()

    with open(args.json_path, "r", encoding="utf-8") as f:
        docs = json.load(f)

    print(f"========== Diversity Check ({args.json_path}) ==========")
    print(f"Total docs: {len(docs)}\n")

    failures = []
    warnings = []

    check_wikipedia_diversity(docs, failures, warnings)
    check_template_leaks(docs, failures, warnings)
    check_restaurant_content_diversity(docs, failures, warnings)
    check_title_suffix_diversity(docs, failures, warnings)
    check_source_url_consistency(docs, failures, warnings)

    print("========== Summary ==========")
    if failures:
        print(f"\n[FAIL] {len(failures)} blocker(s) - corpus 不可入库:")
        for msg in failures:
            print(f"   - {msg}")
    else:
        print("\n[OK] 所有 diversity 检查通过")

    if warnings:
        print(f"\n[WARN] {len(warnings)} warning(s) - 可入库但建议关注:")
        for msg in warnings:
            print(f"   - {msg}")

    sys.exit(1 if failures else 0)


def check_wikipedia_diversity(docs, failures, warnings):
    print("=== [1] Wikipedia source diversity ===")
    wiki_docs = [d for d in docs if d.get("source") == "wikipedia"]
    print(f"Wikipedia 来源文档总数: {len(wiki_docs)}")

    if not wiki_docs:
        warnings.append("没有 source=wikipedia 的文档,跳过 Wikipedia 多样性检查")
        print()
        return

    url_counts = Counter(d["url"] for d in wiki_docs)
    unique_urls = len(url_counts)
    most_chunked = url_counts.most_common(5)

    print(f"Unique URLs: {unique_urls}")
    print("Top 5 most-chunked articles:")
    for url, n in most_chunked:
        article = url.rsplit("/", 1)[-1]
        print(f"  {n:3d} x {article}")

    if unique_urls < MIN_UNIQUE_WIKI_URLS:
        failures.append(
            f"Wikipedia unique URL 数过少: {unique_urls} < {MIN_UNIQUE_WIKI_URLS}"
        )

    if most_chunked:
        top_url, top_n = most_chunked[0]
        top_article = top_url.rsplit("/", 1)[-1]
        if top_n > MAX_CHUNKS_PER_WIKI_URL:
            failures.append(
                f"单篇 Wikipedia chunk 过多: {top_article} = {top_n} 篇 (> {MAX_CHUNKS_PER_WIKI_URL})"
            )
    print()


def check_template_leaks(docs, failures, warnings):
    print("=== [2] 模板痕迹 / meta-phrase 泄漏 ===")
    for phrase in FORBIDDEN_PHRASES:
        hits = sum(1 for d in docs if phrase in d.get("content", ""))
        if hits:
            failures.append(f'"{phrase}" 出现在 {hits} 篇 content (应为 0)')
            print(f"  [FAIL] '{phrase}' 出现 {hits} 次")
        else:
            print(f"  [OK] '{phrase}' 出现 0 次")

    for phrase in WARN_PHRASES:
        hits = sum(
            1 for d in docs if phrase.lower() in d.get("content", "").lower()
        )
        if hits:
            warnings.append(
                f'"{phrase}" 出现在 {hits} 篇 content (核查是否模板残留)'
            )
            print(f"  [WARN] '{phrase}' 出现 {hits} 次")
    print()


def check_restaurant_content_diversity(docs, failures, warnings):
    print("=== [3] Restaurant content template diversity ===")
    restaurants = [d for d in docs if d.get("topic") == "restaurant"]
    print(f"Restaurant 文档总数: {len(restaurants)}")

    if not restaurants:
        warnings.append("没有 topic=restaurant 的文档,跳过模板多样性检查")
        print()
        return

    tails = Counter(
        " ".join(d.get("content", "").split()[-20:]) for d in restaurants
    )
    distinct_tails = len(tails)
    most_common_tail, most_common_n = tails.most_common(1)[0]

    print(f"Distinct tails (last 20 words): {distinct_tails}")
    print(f"Most common tail 被 {most_common_n}/{len(restaurants)} 篇共用")
    print(f"  '...{most_common_tail[:80]}...'")

    if distinct_tails < MIN_DISTINCT_RESTAURANT_TAILS:
        failures.append(
            f"Restaurant 模板单一: distinct tails = {distinct_tails} < {MIN_DISTINCT_RESTAURANT_TAILS}"
        )
    print()


def check_title_suffix_diversity(docs, failures, warnings):
    print("=== [4] Title suffix diversity (按 topic 分组) ===")

    for topic, suffix_len in [("restaurant", 3), ("university", 4)]:
        topic_docs = [d for d in docs if d.get("topic") == topic]
        if not topic_docs:
            continue

        suffixes = Counter(
            " ".join(d.get("title", "").split()[-suffix_len:])
            for d in topic_docs
        )
        most_common_suffix, most_common_n = suffixes.most_common(1)[0]
        pct = 100 * most_common_n / len(topic_docs)

        print(
            f"  {topic} (last {suffix_len} words): "
            f"'{most_common_suffix}' = {most_common_n}/{len(topic_docs)} ({pct:.0f}%)"
        )

        if pct > MAX_TOPIC_SUFFIX_DOMINANCE_PCT:
            failures.append(
                f"{topic} title 模板痕迹: {pct:.0f}% 文档以 "
                f"'{most_common_suffix}' 结尾 (> {MAX_TOPIC_SUFFIX_DOMINANCE_PCT}%)"
            )
    print()


def check_source_url_consistency(docs, failures, warnings):
    print("=== [5] Source / URL consistency ===")

    tripadvisor_docs = [d for d in docs if d.get("source") == "tripadvisor"]
    if not tripadvisor_docs:
        print("  (无 source=tripadvisor 文档,跳过)\n")
        return

    search_url_docs = [
        d for d in tripadvisor_docs if "Search?q=" in d.get("url", "")
    ]
    print(
        f"  source=tripadvisor 文档总数: {len(tripadvisor_docs)}, "
        f"其中 URL 是搜索链接的: {len(search_url_docs)}"
    )

    if search_url_docs:
        warnings.append(
            f"source='tripadvisor' 但 {len(search_url_docs)} 篇 URL 是搜索链接 "
            "(content 非真实评论,建议 source 改为 'synthesized_restaurant')"
        )
    print()


if __name__ == "__main__":
    main()

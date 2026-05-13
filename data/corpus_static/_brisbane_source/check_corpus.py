import json
import re
from collections import Counter


FILE_NAME = "brisbane_corpus.json"

REQUIRED_FIELDS = ["doc_id", "title", "content", "source", "topic", "url"]

ALLOWED_SOURCES = {
    "wikipedia",
    "tripadvisor",
    "tourism_au",
    "uq_official",
    "news",
    "local_curated"
}

ALLOWED_TOPICS = {
    "tourism",
    "restaurant",
    "university",
    "transport",
    "culture_food",
}


def is_valid_doc_id(doc_id):
    return re.match(r"^brisbane_\d{3}$", doc_id) is not None


def main():
    with open(FILE_NAME, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("========== Corpus Check ==========")
    print(f"Total documents: {len(data)}")
    print(f"Unique doc_ids: {len(set(d['doc_id'] for d in data))}")
    print("Topic distribution:", Counter(d["topic"] for d in data))
    print("Source distribution:", Counter(d["source"] for d in data))

    lengths = [len(d["content"].split()) for d in data]

    if lengths:
        print(
            f"Content length: shortest {min(lengths)}, "
            f"longest {max(lengths)}, "
            f"average {sum(lengths) / len(lengths):.0f}"
        )

    print("\n========== Detailed Issues ==========")

    has_issue = False

    doc_ids = [d.get("doc_id", "") for d in data]
    duplicated_ids = [doc_id for doc_id, count in Counter(doc_ids).items() if count > 1]

    if duplicated_ids:
        has_issue = True
        print(f"❌ Duplicated doc_ids: {duplicated_ids}")

    for d in data:
        doc_id = d.get("doc_id", "?")

        missing = [field for field in REQUIRED_FIELDS if field not in d]
        if missing:
            has_issue = True
            print(f"❌ {doc_id} missing fields: {missing}")
            continue

        if not is_valid_doc_id(d["doc_id"]):
            has_issue = True
            print(f"❌ {doc_id} invalid doc_id format")

        if d["source"] not in ALLOWED_SOURCES:
            has_issue = True
            print(f"❌ {doc_id} invalid source: {d['source']}")

        if d["topic"] not in ALLOWED_TOPICS:
            has_issue = True
            print(f"❌ {doc_id} invalid topic: {d['topic']}")

        word_count = len(d["content"].split())
        if word_count < 100 or word_count > 300:
            has_issue = True
            print(f"⚠️ {doc_id} content length is {word_count} words")

        title_word_count = len(d["title"].split())
        if title_word_count < 5 or title_word_count > 15:
            has_issue = True
            print(f"⚠️ {doc_id} title length is {title_word_count} words")

        if d["url"] and not d["url"].startswith("http"):
            has_issue = True
            print(f"⚠️ {doc_id} url may be invalid: {d['url']}")

    print("\n========== Requirement Check ==========")

    topic_counter = Counter(d["topic"] for d in data)

    if 200 <= len(data) <= 300:
        print("✅ Document count is within 200-300")
    else:
        print(f"⚠️ Document count should be between 200 and 300. Current: {len(data)}")

    if len(set(doc_ids)) == len(data):
        print("✅ All doc_ids are unique")
    else:
        print("❌ Some doc_ids are duplicated")

    checks = [
        ("restaurant",  70, 80),
        ("tourism",     70, 80),
        ("university",  50, 60),
        ("transport",   30, 40),
        ("culture_food",30, 40),
    ]
    for topic, lo, hi in checks:
        n = topic_counter.get(topic, 0)
        if n >= lo:
            print(f"✅ {topic}: {n} documents (target {lo}-{hi})")
        else:
            print(f"⚠️ {topic}: only {n} documents (need at least {lo})")


    print("\n========== Diversity Check ==========")

    wiki_docs = [d for d in data if d["source"] == "wikipedia"]
    wiki_url_counts = Counter(d["url"] for d in wiki_docs)
    unique_wiki_urls = len(wiki_url_counts)
    max_chunks = max(wiki_url_counts.values()) if wiki_url_counts else 0
    print(f"Unique Wikipedia URLs: {unique_wiki_urls} (target ≥ 30)")
    print(f"Max chunks from single article: {max_chunks} (target ≤ 3)")
    print(f"Top 5 Wikipedia sources: {wiki_url_counts.most_common(5)}")
    if unique_wiki_urls < 30:
        has_issue = True
        print(f"❌ Wikipedia diversity too low: only {unique_wiki_urls} unique URLs")
    if max_chunks > 3:
        has_issue = True
        print(f"❌ Single Wikipedia article contributes {max_chunks} chunks (limit 3)")

    rag_hits = sum(1 for d in data if "RAG knowledge base" in d.get("content", ""))
    print(f"\nDocs containing 'RAG knowledge base': {rag_hits} (target 0)")
    if rag_hits > 0:
        has_issue = True
        print(f"❌ Found 'RAG knowledge base' in {rag_hits} document(s)")

    restaurant_docs = [d for d in data if d["topic"] == "restaurant"]
    if restaurant_docs:
        tails = [" ".join(d["content"].split()[-20:]) for d in restaurant_docs]
        distinct_tails = len(set(tails))
        print(f"\nRestaurant tail diversity (last 20 words): {distinct_tails} distinct / {len(restaurant_docs)} docs (target ≥ 4)")
        if distinct_tails < 4:
            has_issue = True
            print(f"❌ Restaurant content tails too uniform ({distinct_tails} distinct)")

    title_suffixes_rest = [" ".join(d["title"].split()[-3:]) for d in restaurant_docs]
    rest_suffix_counter = Counter(title_suffixes_rest)
    most_common_rest_suffix, most_common_rest_count = rest_suffix_counter.most_common(1)[0] if rest_suffix_counter else ("", 0)
    print(f"Most common restaurant title suffix: '{most_common_rest_suffix}' × {most_common_rest_count}/{len(restaurant_docs)}")
    if most_common_rest_count == len(restaurant_docs) and len(restaurant_docs) > 0:
        has_issue = True
        print("❌ All restaurant titles share the same suffix")

    if not has_issue:
        print("\n✅ No detailed format issues found.")
    else:
        print("\n⚠️ Some issues need to be fixed.")


if __name__ == "__main__":
    main()
"""
Corpus data structures and JSON loaders.
语料库数据结构与 JSON 加载器。

Document JSON schema:
{
  "doc_id":  "brisbane_001",
  "title":   "...",
  "content": "...",
  "source":  "...",
  "topic":   "...",
  "url":     "..."
}
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import json


@dataclass
class Document:
    """
    A single corpus document. Fields are strings unless noted.
    单篇语料文档,字段除特别说明外为字符串。
    """
    doc_id: str
    title: str
    content: str
    source: str = "unknown"
    topic: str = "general"
    url: str = ""
    # Runtime-only poison flag; not serialised to JSON.
    # 运行时 poison 标记,不写出 JSON。
    is_poison: bool = field(default=False, repr=False)

    @property
    def text_for_embedding(self) -> str:
        """
        Title + content concatenation used as the embedder input.
        embedding 输入文本:title + content 拼接。
        """
        return f"{self.title}\n\n{self.content}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("is_poison", None)  # do not persist runtime flag
        return d


def load_corpus(json_path: Path) -> List[Document]:
    """
    Load a corpus JSON file into a list of Document.
    从 JSON 加载语料库为 Document 列表。

    The JSON must be a list of dicts. Required fields: doc_id, title, content.
    Other fields fall back to defaults defined on Document.

    JSON 顶层是数组,每个元素是 dict。必填 doc_id / title / content;其余走默认值。

    Raises:
        FileNotFoundError: path does not exist.
        ValueError: malformed JSON, missing required fields, or duplicate doc_ids.
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Corpus JSON must be a list, got {type(data).__name__}")

    docs = []
    required = {"doc_id", "title", "content"}
    for i, item in enumerate(data):
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Document at index {i} missing fields: {missing}")
        docs.append(Document(
            doc_id=item["doc_id"],
            title=item["title"],
            content=item["content"],
            source=item.get("source", "unknown"),
            topic=item.get("topic", "general"),
            url=item.get("url", ""),
        ))

    # Reject duplicate doc_ids.
    # doc_id 唯一性检查。
    ids = [d.doc_id for d in docs]
    if len(ids) != len(set(ids)):
        from collections import Counter
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        raise ValueError(f"Duplicate doc_ids: {dupes[:5]}")

    return docs


def load_poison_set(json_path: Path) -> List[Document]:
    """
    Load a poison-set JSON into Document with is_poison=True.
    加载 poison 集合,is_poison 置 True。

    Poison JSON carries extra fields (query_id, attack_type) beyond the corpus
    schema; we stash them into source / topic so the rest of the code can stay
    schema-agnostic.

    Poison JSON 比 corpus 多 query_id / attack_type 两个字段,塞进 source / topic,
    让下游沿用 Document 类型即可。
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Poison file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for i, item in enumerate(data):
        # Poison records only require doc_id + content; the rest is optional.
        # poison 记录至少要有 doc_id + content,其余可选。
        if "doc_id" not in item or "content" not in item:
            raise ValueError(f"Poison doc at index {i} missing doc_id or content")
        doc = Document(
            doc_id=item["doc_id"],
            title=item.get("title", "[poison]"),
            content=item["content"],
            source=f"poison:{item.get('attack_type', 'unknown')}",
            topic=item.get("query_id", "all"),
            url="",
        )
        doc.is_poison = True
        docs.append(doc)
    return docs


def save_corpus(docs: List[Document], json_path: Path):
    """
    Persist a list of Document back to JSON.
    把 Document 列表写回 JSON。
    """
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([d.to_dict() for d in docs], f, ensure_ascii=False, indent=2)

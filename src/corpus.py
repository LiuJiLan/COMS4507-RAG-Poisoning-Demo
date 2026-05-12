"""
语料库数据结构和加载器。

Document schema 对应任务 A 的 JSON 格式：
{
  "doc_id": "brisbane_001",
  "title": "...",
  "content": "...",
  "source": "...",
  "topic": "...",
  "url": "..."
}
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional
import json


@dataclass
class Document:
    """单篇文档。所有字段都是字符串（除非另说）。"""
    doc_id: str
    title: str
    content: str
    source: str = "unknown"
    topic: str = "general"
    url: str = ""
    # 标记是否为 poison 文档（运行时打标，不存在 JSON 里）
    is_poison: bool = field(default=False, repr=False)

    @property
    def text_for_embedding(self) -> str:
        """用于做 embedding 的文本。标题 + 正文拼接。"""
        return f"{self.title}\n\n{self.content}"

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("is_poison", None)  # 不写回文件
        return d


def load_corpus(json_path: Path) -> List[Document]:
    """
    从 JSON 文件加载语料库。
    
    JSON 格式必须是一个数组，每个元素是 Document 的 dict 表示。
    
    Args:
        json_path: JSON 文件路径
    
    Returns:
        Document 列表
    
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: JSON 格式错误或字段缺失
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

    # 检查 doc_id 唯一性
    ids = [d.doc_id for d in docs]
    if len(ids) != len(set(ids)):
        from collections import Counter
        dupes = [k for k, v in Counter(ids).items() if v > 1]
        raise ValueError(f"Duplicate doc_ids: {dupes[:5]}")

    return docs


def load_poison_set(json_path: Path) -> List[Document]:
    """
    加载一组 poison 文档。
    
    Poison JSON 格式比 corpus 多两个字段：query_id, attack_type。
    但这里我们也用 Document 表示，把那两个字段塞到 source/topic 里。
    """
    json_path = Path(json_path)
    if not json_path.exists():
        raise FileNotFoundError(f"Poison file not found: {json_path}")

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = []
    for i, item in enumerate(data):
        # poison 文档至少要有 doc_id 和 content
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
    """保存语料库到 JSON（用于 dummy 数据生成、缓存等）"""
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump([d.to_dict() for d in docs], f, ensure_ascii=False, indent=2)

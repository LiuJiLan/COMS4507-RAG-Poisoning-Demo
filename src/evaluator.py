"""
评估器。核心功能：对比"clean retrieval"和"poisoned retrieval"，
计算 rank 变化的指标。
"""
from dataclasses import dataclass
from typing import List, Optional
from .retriever import RetrievalResult


@dataclass
class AttackMetrics:
    """单次攻击的评估指标"""
    poison_in_topk: bool         # poison 是否进入 top-k
    poison_rank: Optional[int]   # poison 的具体排名（None 表示未进 top-k）
    n_poison_in_topk: int        # 进入 top-k 的 poison 数（可能多个）
    displaced_docs: List[str]    # 被挤出 top-k 的原始文档 ID 列表
    score_gap: Optional[float]   # poison 分数 - clean top-1 分数（用于看"挤压程度"）

    def __repr__(self):
        if self.poison_in_topk:
            return (f"AttackMetrics(SUCCESS, poison_rank={self.poison_rank}, "
                    f"displaced={len(self.displaced_docs)})")
        return "AttackMetrics(FAIL, poison not in top-k)"


def compare_rankings(clean_results: List[RetrievalResult],
                     poisoned_results: List[RetrievalResult]) -> AttackMetrics:
    """
    对比 clean 和 poisoned 两次检索结果，计算攻击指标。
    
    Args:
        clean_results: 干净知识库的 top-k
        poisoned_results: 注入 poison 后的 top-k
    
    Returns:
        AttackMetrics
    """
    # 找出 poisoned 里的 poison 文档
    poison_results = [r for r in poisoned_results if r.doc.is_poison]
    poison_in_topk = len(poison_results) > 0

    poison_rank = None
    if poison_in_topk:
        poison_rank = min(r.rank for r in poison_results)

    # 哪些原本在 top-k 的文档被挤出去了
    clean_ids = set(r.doc.doc_id for r in clean_results)
    poisoned_ids = set(r.doc.doc_id for r in poisoned_results)
    displaced = clean_ids - poisoned_ids

    # 分数差距
    score_gap = None
    if poison_in_topk and clean_results:
        best_poison_score = max(r.score for r in poison_results)
        clean_top_score = clean_results[0].score
        score_gap = float(best_poison_score - clean_top_score)

    return AttackMetrics(
        poison_in_topk=poison_in_topk,
        poison_rank=poison_rank,
        n_poison_in_topk=len(poison_results),
        displaced_docs=list(displaced),
        score_gap=score_gap,
    )


def format_ranking_table(results: List[RetrievalResult],
                         max_title_len: int = 40) -> str:
    """把检索结果格式化成文本表格（用于日志/控制台输出）"""
    if not results:
        return "(no results)"
    lines = []
    for r in results:
        marker = "☣" if r.doc.is_poison else " "
        title = r.doc.title[:max_title_len]
        lines.append(f"  {r.rank:>2}. {marker} {title:<{max_title_len}}  {r.score:.4f}")
    return "\n".join(lines)

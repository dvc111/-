"""Top-K 候选答案筛选。

接收概率向量和主题实体 ID 列表，
排除主题实体后，选取概率最高的 K 个节点作为候选答案。
宏观和微观的 inference.py 都会调用此模块。"""

from __future__ import annotations

import torch
from typing import List, Optional, Tuple, Dict, Any


def filter_top_k(
    scores: torch.Tensor,
    node_ids: List[str],
    topic_entity_ids: Optional[List[str]] = None,
    top_k: int = 10,
) -> List[Dict[str, Any]]:
    """从概率向量中筛选 Top-K 候选答案，排除主题实体。"""
    N = scores.size(0)
    topic_set = set(topic_entity_ids) if topic_entity_ids else set()

    candidates: List[Tuple[int, float]] = []
    for i in range(N):
        if node_ids[i] not in topic_set:
            candidates.append((i, float(scores[i])))

    candidates.sort(key=lambda x: x[1], reverse=True)

    return [
        {"entity_id": node_ids[idx], "prob": score}
        for idx, score in candidates[:top_k]
    ]
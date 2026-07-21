"""宏观特征构建：BERT 语义嵌入 + is_topic 标志位。

取每个实体的 BERT 语义嵌入，追加 is_topic 标志位，
标记哪些节点是问题中的主题实体（推理起点），不包含 DDE。
宏观的创新点是"超关系聚合 + 一次规划"，
通过 is_topic 告诉 GNN"从这里开始推理"。
"""

from __future__ import annotations

import torch


def build_macro_features(
    bert_embeddings: torch.Tensor,
    entity_ids: list[str],
    topic_entity_ids: list[str],
) -> torch.Tensor:
    """拼接 BERT 语义嵌入与 is_topic 标志位，形成宏观节点特征。

    is_topic 位表示该实体是否为问题中的主题实体（推理起点），
    用于在 R-GCN 前向传播中给 GNN 提供"从哪开始推理"的结构提示。

    Args:
        bert_embeddings:   BERT 文本嵌入, shape (N, bert_dim)。
        entity_ids:        与 bert_embeddings 行对应的实体 ID 列表，长度 N。
        topic_entity_ids:  问题中的主题实体 ID 列表（推理起点集合）。

    Returns:
        宏观特征矩阵, shape (N, bert_dim + 1)。
        每行最后一列为 is_topic: 1.0 表示主题实体，0.0 表示非主题实体。
    """
    N = bert_embeddings.size(0)
    if len(entity_ids) != N:
        raise ValueError(
            f"entity_ids 长度 ({len(entity_ids)}) 与 "
            f"bert_embeddings 行数 ({N}) 不匹配"
        )

    topic_set = set(topic_entity_ids)
    is_topic = torch.zeros(N, 1)
    for i, eid in enumerate(entity_ids):
        if eid in topic_set:
            is_topic[i] = 1.0

    return torch.cat([bert_embeddings, is_topic], dim=-1)


def build_macro_features_from_dict(
    bert_dict: dict[str, torch.Tensor],
    entity_ids: list[str],
    topic_entity_ids: list[str],
) -> torch.Tensor:
    """从字典按实体 ID 顺序组装宏观特征。

    Args:
        bert_dict:         实体 ID → BERT 嵌入向量。
        entity_ids:        有序的实体 ID 列表，决定矩阵的行顺序。
        topic_entity_ids:  主题实体 ID 列表。

    Returns:
        宏观特征矩阵, shape (N, bert_dim + 1)。
    """
    emb_list = []
    for eid in entity_ids:
        emb_list.append(bert_dict[eid].view(1, -1))

    bert_mat = torch.cat(emb_list, dim=0)
    return build_macro_features(bert_mat, entity_ids, topic_entity_ids)
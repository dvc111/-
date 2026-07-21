"""微观特征构建：BERT 语义嵌入 + DDE 有向距离编码联合建模。

取每个实体的 BERT 语义嵌入，拼接上外部传入的 DDE 矩阵（64 维），
形成 [N, bert_dim + 64] 的联合特征，供微观侧 R-GCN 作为节点初始特征输入。
微观的核心创新是"DDE + 语义联合建模"——两套软著最大的代码差异点。
"""

from __future__ import annotations

import torch
from typing import Optional


def build_micro_features(
    bert_embeddings: torch.Tensor,
    dde_matrix: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """拼接 BERT 语义嵌入与 DDE 结构编码，形成微观联合特征。

    Args:
        bert_embeddings:  BERT 文本嵌入, shape (N, bert_dim)。
        dde_matrix:       DDE 矩阵, shape (N, 64)。为 None 时只返回 bert_embeddings。

    Returns:
        联合特征矩阵, shape (N, bert_dim + 64) 或 (N, bert_dim)。
    """
    if dde_matrix is None:
        return bert_embeddings

    if bert_embeddings.size(0) != dde_matrix.size(0):
        raise ValueError(
            f"BERT 行数 ({bert_embeddings.size(0)}) 与 "
            f"DDE 行数 ({dde_matrix.size(0)}) 不匹配"
        )

    return torch.cat([bert_embeddings, dde_matrix], dim=-1)


def build_micro_features_from_dict(
    bert_dict: dict[str, torch.Tensor],
    dde_dict: dict[str, torch.Tensor],
    entity_ids: list[str],
) -> torch.Tensor:
    """从字典按实体 ID 顺序组装并拼接 BERT + DDE 特征。

    Args:
        bert_dict:  实体 ID → BERT 嵌入。
        dde_dict:   实体 ID → DDE 向量（64 维）。
        entity_ids: 有序的实体 ID 列表。

    Returns:
        联合特征矩阵, shape (N, bert_dim + 64)。
    """
    bert_list = [bert_dict[eid].view(1, -1) for eid in entity_ids]
    dde_list = [dde_dict.get(eid, torch.zeros(64)).view(1, -1) for eid in entity_ids]
    return build_micro_features(torch.cat(bert_list, dim=0), torch.cat(dde_list, dim=0))
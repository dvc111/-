"""微观数据装配。

把微观证据子图 + DDE 矩阵转成统一的 GraphData 格式，
调用 features.py 生成 BERT + DDE 节点特征。
微观的输入比宏观多一个 DDE 矩阵，需要单独处理。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

_BACKEND_DIR = str(Path(__file__).resolve().parents[2])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from gnn.core.base_loader import triples_to_graph_data, GraphData
from gnn.micro_gnn.features import build_micro_features


def assemble_micro_subgraph(
    triples: list[tuple[str, str, str]],
    entity_embeddings: dict[str, torch.Tensor],
    entity_dde: dict[str, torch.Tensor],
    relation_to_id: Optional[dict[str, int]] = None,
) -> GraphData:
    """将微观证据子图 + DDE 装配为带特征的 GraphData。

    流程: triples_to_graph_data → build_micro_features → 写入 node_features

    Args:
        triples:            证据三元组列表，(head_id, relation_id, tail_id)。
        entity_embeddings:  实体 ID → BERT 嵌入。
        entity_dde:         实体 ID → DDE 向量（64 维）。
        relation_to_id:     关系 → 整数 ID 映射（可选）。

    Returns:
        装配完成的 GraphData:
            node_features: (N, bert_dim + 64)，BERT + DDE
            edge_index:    (2, E)
            edge_type:     (E,)
            node_ids:      实体 ID 列表
    """
    graph_data = triples_to_graph_data(triples, relation_to_id=relation_to_id)

    bert_list = []
    dde_list = []
    for eid in graph_data.node_ids:
        bert_list.append(entity_embeddings.get(eid, torch.zeros(768)).view(1, -1))
        dde_list.append(entity_dde.get(eid, torch.zeros(64)).view(1, -1))

    joint = build_micro_features(
        torch.cat(bert_list, dim=0),
        torch.cat(dde_list, dim=0),
    )
    graph_data.node_features = joint
    return graph_data
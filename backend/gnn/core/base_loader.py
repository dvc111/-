"""GraphData + 三元组转图。GNN 模块统一数据格式。"""

from __future__ import annotations
import torch
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class GraphData:
    """子图数据容器，统一存放特征矩阵、边索引和节点映射。"""
    node_features: torch.Tensor          # (N, F) 节点特征
    edge_index: torch.Tensor             # (2, E) 边索引，第0行起点第1行终点
    edge_type: torch.Tensor              # (E,)   每条边的关系类型编号
    node_ids: List[str] = field(default_factory=list)
    node_id_to_idx: dict = field(default_factory=dict)

    @property
    def num_nodes(self):
        return self.node_features.size(0)

    @property
    def num_edges(self):
        return self.edge_index.size(1)

    @property
    def num_relations(self):
        if self.edge_type.numel() == 0:
            return 0
        return int(self.edge_type.max().item()) + 1


def triples_to_graph_data(triples, relation_to_id=None):
    """三元组列表 → GraphData（实体去重、建边、标关系类型）。"""
    nodes = sorted({e for t in triples for e in [t[0], t[2]]})
    idx = {e: i for i, e in enumerate(nodes)}

    if relation_to_id is None:
        rs = sorted({r for _, r, _ in triples})
        relation_to_id = {r: i for i, r in enumerate(rs)}

    E = len(triples)
    ei = torch.zeros((2, E), dtype=torch.long)
    et = torch.zeros(E, dtype=torch.long)
    for i, (h, r, t) in enumerate(triples):
        ei[0, i], ei[1, i] = idx[h], idx[t]
        et[i] = relation_to_id[r]

    return GraphData(node_features=torch.zeros(len(nodes), 128), edge_index=ei, edge_type=et, node_ids=nodes, node_id_to_idx=idx)
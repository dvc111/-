"""
通用 PyG 数据转换工具

将宏观检索 / 微观检索产出的结构化数据（三元组列表、节点特征、DDE 等）
转换为 PyTorch Geometric 的 Data 对象，供 R-GCN 训练和推理使用。
"""

from __future__ import annotations

import torch
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class GraphData:
    """轻量图数据容器，供 R-GCN 前向传播使用。

    宏观和微观模块在检索阶段产出后，统一转成此结构再喂给模型。
    """

    node_features: torch.Tensor          # (N, feat_dim)
    edge_index: torch.Tensor             # (2, E), long
    edge_type: torch.Tensor              # (E,), long, [0, num_relations-1]
    node_ids: List[str] = field(default_factory=list)       # 实体 ID 列表
    node_id_to_idx: Dict[str, int] = field(default_factory=dict)  # ID → 行号

    @property
    def num_nodes(self) -> int:
        return self.node_features.size(0)

    @property
    def num_edges(self) -> int:
        return self.edge_index.size(1)

    @property
    def num_relations(self) -> int:
        if self.edge_type.numel() == 0:
            return 0
        return int(self.edge_type.max().item()) + 1

    def to(self, device: torch.device) -> GraphData:
        self.node_features = self.node_features.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_type = self.edge_type.to(device)
        return self

    def to_pyg_data(self):
        """转换为 PyG Data 对象，方便接入 PyG 的数据处理流程。"""
        try:
            from torch_geometric.data import Data
            return Data(
                x=self.node_features,
                edge_index=self.edge_index,
                edge_type=self.edge_type,
            )
        except ImportError:
            raise ImportError(
                "torch_geometric is required for to_pyg_data(). "
                "Install it via: pip install torch_geometric"
            )


def triples_to_graph_data(
    triples: List[Tuple[str, str, str]],
    node_features: Optional[Dict[str, torch.Tensor]] = None,
    default_feature_dim: int = 128,
    relation_to_id: Optional[Dict[str, int]] = None,
) -> GraphData:
    """将三元组列表转换为 GraphData。

    流程：
    1. 收集所有出现过的实体 ID，为其分配连续索引；
    2. 为每个实体构建特征向量（从 node_features 中取，缺失则用零向量）；
    3. 为每条三元组建立边索引（head → tail）和关系类型 ID。

    Args:
        triples:          三元组列表，每个元素 (head_id, relation_id, tail_id)
        node_features:    可选，实体 ID → 特征向量的映射
        default_feature_dim: 默认特征维度（当 node_features 为空时）
        relation_to_id:   可选，关系 → 整数 ID 的映射；未传入时自动构建

    Returns:
        GraphData 对象
    """
    # 收集所有实体
    entity_set: set = set()
    for h, r, t in triples:
        entity_set.add(h)
        entity_set.add(t)
    node_ids = sorted(entity_set)
    node_id_to_idx = {eid: i for i, eid in enumerate(node_ids)}
    N = len(node_ids)
    E = len(triples)

    # 建立边索引
    edge_index = torch.zeros((2, E), dtype=torch.long)
    for i, (h, r, t) in enumerate(triples):
        edge_index[0, i] = node_id_to_idx[h]
        edge_index[1, i] = node_id_to_idx[t]

    # 建立关系类型 ID
    if relation_to_id is None:
        rel_set = sorted({r for _, r, _ in triples})
        relation_to_id = {rid: i for i, rid in enumerate(rel_set)}
    edge_type = torch.tensor(
        [relation_to_id[r] for _, r, _ in triples],
        dtype=torch.long,
    )

    # 构建节点特征矩阵
    if node_features is not None and len(node_features) > 0:
        sample_feat = next(iter(node_features.values()))
        feat_dim = sample_feat.size(-1)
        feat_mat = torch.zeros(N, feat_dim)
        for eid, idx in node_id_to_idx.items():
            if eid in node_features:
                feat_mat[idx] = node_features[eid].view(-1)
    else:
        feat_mat = torch.zeros(N, default_feature_dim)

    return GraphData(
        node_features=feat_mat,
        edge_index=edge_index,
        edge_type=edge_type,
        node_ids=node_ids,
        node_id_to_idx=node_id_to_idx,
    )


def build_node_feature_matrix(
    node_embeddings: Dict[str, torch.Tensor],
    node_ids: List[str],
    default_dim: int = 128,
) -> torch.Tensor:
    """根据实体 ID 列表从嵌入字典中按序组装特征矩阵。

    Args:
        node_embeddings:  实体 ID → 嵌入向量
        node_ids:         有序的实体 ID 列表，决定矩阵的行顺序
        default_dim:      默认特征维度，当嵌入字典为空时使用

    Returns:
        特征矩阵, shape (len(node_ids), feat_dim)
    """
    if not node_embeddings:
        return torch.zeros(len(node_ids), default_dim)

    sample = next(iter(node_embeddings.values()))
    dim = sample.size(-1)
    mat = torch.zeros(len(node_ids), dim)
    for i, eid in enumerate(node_ids):
        if eid in node_embeddings:
            mat[i] = node_embeddings[eid].view(-1)
    return mat


def dde_collate(
    evidence_triples: List[Dict],
    entity_dde: Dict[str, torch.Tensor],
    node_embeddings: Dict[str, torch.Tensor],
    relation_to_id: Optional[Dict[str, int]] = None,
) -> GraphData:
    """微观检索产出的 DDE 增强证据子图 → GraphData。

    每个 evidence_triple 包含三元组、相关分数和 DDE 信息，
    拼接 DDE 编码和文本嵌入作为节点初始特征。

    Args:
        evidence_triples: 微观检索结果，格式见接口文档
        entity_dde:       实体 ID → DDE 向量
        node_embeddings:  实体 ID → BERT 文本嵌入
        relation_to_id:   可选，关系 → 整数 ID

    Returns:
        GraphData 对象
    """
    triples = []
    for et in evidence_triples:
        triples.append(tuple(et["triple"]))

    base = triples_to_graph_data(triples, relation_to_id=relation_to_id)

    # 拼接 DDE + 文本嵌入作为节点特征
    feat_dim = 0
    if node_embeddings:
        feat_dim += next(iter(node_embeddings.values())).size(-1)
    if entity_dde:
        feat_dim += next(iter(entity_dde.values())).size(-1)
    if feat_dim == 0:
        feat_dim = 128

    feat_mat = torch.zeros(base.num_nodes, feat_dim)
    for eid, idx in base.node_id_to_idx.items():
        parts = []
        if node_embeddings and eid in node_embeddings:
            parts.append(node_embeddings[eid].view(-1))
        if entity_dde and eid in entity_dde:
            parts.append(entity_dde[eid].view(-1))
        if parts:
            feat_mat[idx] = torch.cat(parts)

    base.node_features = feat_mat
    return base


def macro_subgraph_to_graph_data(
    triples: List[Tuple[str, str, str]],
    hyper_relation_ids: List[str],
    topic_entity_ids: List[str],
    entity_labels: Dict[str, str],
    entity_embeddings: Optional[Dict[str, torch.Tensor]] = None,
    embed_dim: int = 128,
) -> GraphData:
    """宏观检索产出的超关系约束子图 → GraphData。

    宏观子图的节点初始特征可以使用超关系嵌入或文本嵌入，
    同时标记主题实体信息（可通过 is_topic 特征位或额外特征拼接）。

    Args:
        triples:             三元组列表
        hyper_relation_ids:  筛选后保留的超关系 ID 列表
        topic_entity_ids:    主题实体 ID 列表
        entity_labels:       实体 ID → 标签名
        entity_embeddings:   可选，实体 ID → 嵌入向量
        embed_dim:           默认嵌入维度

    Returns:
        GraphData 对象
    """
    base = triples_to_graph_data(triples, node_features=entity_embeddings,
                                 default_feature_dim=embed_dim)

    # 追加 is_topic 特征位 (1 表示主题实体, 0 表示非主题实体)
    topic_feat = torch.zeros(base.num_nodes, 1)
    for eid in topic_entity_ids:
        if eid in base.node_id_to_idx:
            topic_feat[base.node_id_to_idx[eid]] = 1.0

    base.node_features = torch.cat([base.node_features, topic_feat], dim=-1)
    return base


__all__ = [
    "GraphData",
    "triples_to_graph_data",
    "build_node_feature_matrix",
    "dde_collate",
    "macro_subgraph_to_graph_data",
]

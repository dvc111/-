"""宏观数据装配。

把宏观子图（三元组列表）转成统一的 GraphData 格式，
调用 features.py 生成 BERT + is_topic 节点特征。
模型只认 GraphData，这一步负责"装盘"。
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
from gnn.macro_gnn.features import build_macro_features


def assemble_macro_subgraph(
    triples: list[tuple[str, str, str]],
    entity_embeddings: dict[str, torch.Tensor],
    topic_entity_ids: list[str],
    entity_labels: Optional[dict[str, str]] = None,
    relation_to_id: Optional[dict[str, int]] = None,
) -> GraphData:
    """将宏观子图的三元组列表装配为带特征的 GraphData。

    流程:
        triples_to_graph_data → build_macro_features → 写入 node_features

    Args:
        triples:            宏观子图三元组列表，每项 (head_id, relation_id, tail_id)。
        entity_embeddings:  实体 ID → BERT 嵌入向量（LLM 模块产出）。
        topic_entity_ids:   主题实体 ID 列表（推理起点）。
        entity_labels:      实体 ID → 标签名（可选，存入 GraphData 供后续输出用）。
        relation_to_id:     关系 → 整数 ID 映射（可选，缺失时自动构建）。

    Returns:
        装配完成的 GraphData，内含:
            node_features:   (N, bert_dim + 1)，BERT + is_topic
            edge_index:      (2, E)
            edge_type:       (E,)
            node_ids:        实体 ID 列表
            node_id_to_idx:  实体 ID → 行索引
    """
    # ── 1. 三元组 → 基础 GraphData（不带特征） ──
    graph_data = triples_to_graph_data(
        triples=triples,
        relation_to_id=relation_to_id,
    )

    # ── 2. 构建宏观特征（BERT + is_topic） ──
    bert_list = []
    for eid in graph_data.node_ids:
        if eid in entity_embeddings:
            bert_list.append(entity_embeddings[eid].view(1, -1))
        else:
            # 缺 embedding 时补零向量（罕见情况，保持鲁棒）
            dim = _infer_bert_dim(entity_embeddings)
            bert_list.append(torch.zeros(1, dim))

    bert_embeddings = torch.cat(bert_list, dim=0)

    macro_features = build_macro_features(
        bert_embeddings=bert_embeddings,
        entity_ids=graph_data.node_ids,
        topic_entity_ids=topic_entity_ids,
    )

    # ── 3. 写入特征 ──
    graph_data.node_features = macro_features

    return graph_data


def _infer_bert_dim(entity_embeddings: dict[str, torch.Tensor]) -> int:
    """从嵌入字典中推断 BERT 维度。"""
    if not entity_embeddings:
        return 768  # BERT-base 默认
    return next(iter(entity_embeddings.values())).size(-1)
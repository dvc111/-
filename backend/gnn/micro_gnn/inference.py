"""微观推理入口。

加载 micro_model.pth，接收微观子图 + DDE 矩阵，
调用 scoring 排序和 pathgen 提路径，返回候选答案 + 推理路径。
这是微观软著的对外接口——前端/调度器调这个函数。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

import torch

_BACKEND_DIR = str(Path(__file__).resolve().parents[2])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from gnn.core.base_loader import GraphData
from gnn.micro_gnn.features import build_micro_features
from gnn.scoring.ranker import filter_top_k
from gnn.pathgen.bfs import extract_shortest_path


def run_micro_inference(
    checkpoint_path: str,
    graph_data: GraphData,
    topic_entity_ids: list[str],
    entity_embeddings: dict[str, torch.Tensor],
    entity_dde: dict[str, torch.Tensor],
    entity_labels: Optional[dict[str, str]] = None,
    relation_labels: Optional[dict[str, str]] = None,
    relation_id_map: Optional[dict[int, str]] = None,
    relation_embeddings: Optional[dict] = None,
    question_embedding: Optional[torch.Tensor] = None,
    top_k: int = 10,
    max_hops: int = 3,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """微观推理全流程入口。

    调度流程:
        build_micro_features → load_and_score → filter_top_k → extract_shortest_path

    Args:
        checkpoint_path:   micro_model.pth 路径。
        graph_data:        微观证据子图。
        topic_entity_ids:  主题实体 ID 列表。
        entity_embeddings: 实体 ID → BERT 嵌入。
        entity_dde:        实体 ID → DDE 向量（64 维）。
        entity_labels:     实体 ID → 标签名（可选）。
        relation_labels:   关系 ID → 标签名（可选）。
        relation_id_map:   关系 int → 关系 ID（可选）。
        relation_embeddings: 关系嵌入（可选，路径择优用）。
        question_embedding: 问题嵌入（可选，路径择优用）。
        top_k:              保留候选数，默认 10。
        max_hops:           路径最大跳数，默认 3。
        device:             推理设备。

    Returns:
        {"candidate_answers": [...], "reasoning_paths": [...]}
    """
    # ── 1. 构建微观特征（BERT + DDE） ──
    micro_features = build_micro_features(
        bert_embeddings=torch.stack([entity_embeddings[eid] for eid in graph_data.node_ids]),
        dde_matrix=torch.stack([entity_dde[eid] for eid in graph_data.node_ids]),
    )
    graph_data.node_features = micro_features

    # ── 2. 推理评分 ──
    from gnn.scoring.node_scorer import load_and_score

    scores = load_and_score(
        checkpoint_path=checkpoint_path,
        graph_data=graph_data,
        device=device,
        in_dim=micro_features.size(-1),
    )

    # ── 3. Top-K 筛选 ──
    candidates = filter_top_k(
        scores=scores,
        node_ids=graph_data.node_ids,
        topic_entity_ids=topic_entity_ids,
        top_k=top_k,
    )
    if entity_labels:
        for c in candidates:
            if c["entity_id"] in entity_labels:
                c["label"] = entity_labels[c["entity_id"]]

    # ── 4. 路径提取 ──
    reasoning_paths = []
    for cand in candidates:
        path_result = extract_shortest_path(
            edge_index=graph_data.edge_index,
            edge_type=graph_data.edge_type,
            node_ids=graph_data.node_ids,
            node_id_to_idx=graph_data.node_id_to_idx,
            topic_entity_ids=topic_entity_ids,
            answer_entity_id=cand["entity_id"],
            max_hops=max_hops,
            entity_labels=entity_labels,
            relation_labels=relation_labels,
            relation_embeddings=relation_embeddings,
            question_embedding=question_embedding,
            relation_id_map=relation_id_map,
        )
        if path_result is not None:
            reasoning_paths.append(path_result)

    return {"candidate_answers": candidates, "reasoning_paths": reasoning_paths}
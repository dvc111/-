"""宏观推理入口。

加载 macro_model.pth，接收宏观子图，
调用 scoring 排序和 pathgen 提路径，
返回候选答案 + 推理路径。
这是宏观软著的对外接口——前端/调度器调这个函数。
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
from gnn.macro_gnn.features import build_macro_features
from gnn.scoring.ranker import filter_top_k
from gnn.pathgen.bfs import extract_shortest_path


def run_macro_inference(
    checkpoint_path: str,
    graph_data: GraphData,
    topic_entity_ids: list[str],
    entity_embeddings: dict[str, torch.Tensor],
    entity_labels: Optional[dict[str, str]] = None,
    relation_labels: Optional[dict[str, str]] = None,
    relation_id_map: Optional[dict[int, str]] = None,
    relation_embeddings: Optional[dict] = None,
    question_embedding: Optional[torch.Tensor] = None,
    top_k: int = 10,
    max_hops: int = 3,
    device: Optional[torch.device] = None,
) -> dict[str, Any]:
    """宏观推理全流程入口。

    调度流程:
        build_macro_features → load_and_score → filter_top_k → extract_shortest_path

    Args:
        checkpoint_path:     macro_model.pth 路径。
        graph_data:          宏观子图（含 edge_index, edge_type, node_ids 等）。
        topic_entity_ids:    主题实体 ID 列表（推理起点）。
        entity_embeddings:   实体 ID → BERT 嵌入向量（LLM 模块产出）。
        entity_labels:       实体 ID → 标签名（可选，输出用）。
        relation_labels:     关系 ID → 标签名（可选，输出用）。
        relation_id_map:     关系类型 int → 关系 ID 字符串（可选）。
        relation_embeddings: 关系 ID → 嵌入向量（可选，等长路径择优用）。
        question_embedding:  问题文本嵌入（可选，等长路径择优用）。
        top_k:               保留的候选答案数，默认 10。
        max_hops:            路径最大跳数，默认 3。
        device:              推理设备。

    Returns:
        {
            "candidate_answers": [{"entity_id": str, "prob": float, "label": str?}, ...],
            "reasoning_paths": [{
                "answer_entity_id": str,
                "path": [{"entity_id" or "relation_id": str, "label": str?}, ...],
                "path_score": float,
            }, ...],
        }
    """
    # ── 1. 构建宏观特征（BERT + is_topic） ──
    macro_features = build_macro_features(
        bert_embeddings=torch.stack([
            entity_embeddings[eid] for eid in graph_data.node_ids
        ]),
        entity_ids=graph_data.node_ids,
        topic_entity_ids=topic_entity_ids,
    )

    # 替换 GraphData 中的特征矩阵
    graph_data.node_features = macro_features

    # ── 2. 推理评分（load_and_score） ──
    # 直接从 scoring.node_scorer 导入
    from gnn.scoring.node_scorer import load_and_score

    scores = load_and_score(
        checkpoint_path=checkpoint_path,
        graph_data=graph_data,
        device=device,
        in_dim=macro_features.size(-1),
    )

    # ── 3. Top-K 筛选 ──
    candidates = filter_top_k(
        scores=scores,
        node_ids=graph_data.node_ids,
        topic_entity_ids=topic_entity_ids,
        top_k=top_k,
    )

    # 附上 label（如果有）
    if entity_labels:
        for c in candidates:
            eid = c["entity_id"]
            if eid in entity_labels:
                c["label"] = entity_labels[eid]

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

    return {
        "candidate_answers": candidates,
        "reasoning_paths": reasoning_paths,
    }
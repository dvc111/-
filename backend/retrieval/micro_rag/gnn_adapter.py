"""将微观检索 JSON 转成 GNN 装配所需的数据，不强制依赖 PyTorch。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class GNNHandoff:
    """经过维度和引用校验的微观检索 → GNN 交接对象。"""

    triples: tuple[tuple[str, str, str], ...]
    node_embeddings: dict[str, list[float]]
    entity_dde: dict[str, list[float]]
    entity_labels: dict[str, str]
    relation_labels: dict[str, str]
    relation_to_id: dict[str, int]
    gnn_input_dim: int


def prepare_gnn_handoff(payload: dict[str, Any]) -> GNNHandoff:
    """校验并拆分 `micro_evidence_subgraph`，供 GNN 直接转 Tensor。"""

    if not isinstance(payload, dict):
        raise ValueError("微观检索结果必须是 JSON 对象")
    spec = payload.get("feature_spec")
    if not isinstance(spec, dict):
        raise ValueError("缺少 feature_spec")
    dde_dim = int(spec.get("dde_dim", 0))
    embedding_dim = int(spec.get("text_embedding_dim", 0))
    expected_input_dim = int(spec.get("gnn_input_dim", 0))
    if dde_dim <= 0 or embedding_dim <= 0:
        raise ValueError("DDE 和文本嵌入维度必须大于 0")
    if expected_input_dim != dde_dim + embedding_dim:
        raise ValueError("gnn_input_dim 与文本嵌入、DDE 维度之和不一致")

    node_embeddings: dict[str, list[float]] = {}
    entity_labels: dict[str, str] = {}
    for item in payload.get("node_features", []):
        if not isinstance(item, dict) or not item.get("entity_id"):
            raise ValueError("node_features 中存在无效节点")
        entity_id = str(item["entity_id"])
        vector = item.get("text_embedding")
        if not isinstance(vector, list) or len(vector) != embedding_dim:
            raise ValueError(f"实体 {entity_id} 的文本嵌入维度不正确")
        if entity_id in node_embeddings:
            raise ValueError(f"node_features 中存在重复实体：{entity_id}")
        node_embeddings[entity_id] = [float(value) for value in vector]
        entity_labels[entity_id] = str(item.get("label") or entity_id)

    raw_dde = payload.get("entity_dde")
    if not isinstance(raw_dde, dict):
        raise ValueError("缺少 entity_dde")
    entity_dde: dict[str, list[float]] = {}
    for entity_id, vector in raw_dde.items():
        if not isinstance(vector, list) or len(vector) != dde_dim:
            raise ValueError(f"实体 {entity_id} 的 DDE 维度不正确")
        entity_dde[str(entity_id)] = [float(value) for value in vector]

    triples: list[tuple[str, str, str]] = []
    previous_score = float("inf")
    for item in payload.get("evidence_triples", []):
        if not isinstance(item, dict):
            raise ValueError("evidence_triples 中存在无效证据")
        raw_triple = item.get("triple")
        if not isinstance(raw_triple, list) or len(raw_triple) != 3:
            raise ValueError("证据三元组必须为 [head_id, relation_id, tail_id]")
        head, relation, tail = (str(value) for value in raw_triple)
        if head not in node_embeddings or tail not in node_embeddings:
            raise ValueError("证据三元组引用了 node_features 中不存在的实体")
        if head not in entity_dde or tail not in entity_dde:
            raise ValueError("证据三元组引用了 entity_dde 中不存在的实体")
        score = float(item.get("relevance_score", 0.0))
        if score > previous_score + 1e-12:
            raise ValueError("evidence_triples 未按 relevance_score 降序排列")
        previous_score = score
        triples.append((head, relation, tail))

    relation_labels = {
        str(key): str(value) for key, value in payload.get("relation_labels", {}).items()
    }
    raw_relation_map = payload.get("relation_to_id")
    if isinstance(raw_relation_map, dict):
        relation_to_id = {str(key): int(value) for key, value in raw_relation_map.items()}
    else:
        relation_to_id = {
            relation: index
            for index, relation in enumerate(sorted({item[1] for item in triples}))
        }
    return GNNHandoff(
        triples=tuple(triples),
        node_embeddings=node_embeddings,
        entity_dde=entity_dde,
        entity_labels=entity_labels,
        relation_labels=relation_labels,
        relation_to_id=relation_to_id,
        gnn_input_dim=expected_input_dim,
    )


def prepare_torch_gnn_inputs(payload: dict[str, Any]) -> dict[str, Any]:
    """可选便捷入口：安装 PyTorch 后直接得到 GNN 所需张量字典。"""

    try:
        import torch
    except ImportError as error:  # pragma: no cover - 核心模块无需 torch
        raise RuntimeError("请先安装 GNN 可选依赖：pip install -e .[gnn]") from error
    handoff = prepare_gnn_handoff(payload)
    return {
        "triples": list(handoff.triples),
        "node_embeddings": {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in handoff.node_embeddings.items()
        },
        "entity_dde": {
            key: torch.tensor(value, dtype=torch.float32)
            for key, value in handoff.entity_dde.items()
        },
        "entity_labels": handoff.entity_labels,
        "relation_labels": handoff.relation_labels,
        "relation_to_id": handoff.relation_to_id,
        "gnn_input_dim": handoff.gnn_input_dim,
    }

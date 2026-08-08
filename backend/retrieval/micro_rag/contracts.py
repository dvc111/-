"""团队模块接口 v0.1 的输入解析与输出组装。"""

from __future__ import annotations

from typing import Any

from .dde import directional_distance_encoding
from .models import Triple
from .retriever import MicroRetriever


SCHEMA_VERSION = "0.1"


def _required_string(payload: dict[str, Any], field: str) -> str:
    value = payload.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} 必须是非空字符串")
    return value


def _relation_labels(payload: dict[str, Any], macro_subgraph: dict[str, Any]) -> dict[str, str]:
    result: dict[str, str] = {}
    raw_mapping = payload.get("relation_labels", {})
    if isinstance(raw_mapping, dict):
        result.update({str(key): str(value) for key, value in raw_mapping.items()})
    for source in (
        payload.get("selected_hyper_relations", []),
        macro_subgraph.get("relations", []),
    ):
        if not isinstance(source, list):
            continue
        for item in source:
            if isinstance(item, dict) and item.get("relation_id") and item.get("label"):
                result[str(item["relation_id"])] = str(item["label"])
    return result


def parse_macro_subgraph(payload: dict[str, Any]) -> tuple[str, str, list[str], list[Triple], dict[str, dict[str, Any]]]:
    """将宏观模块 v0.1 JSON 转换为微观检索内部对象。"""

    if not isinstance(payload, dict):
        raise ValueError("请求体必须是 JSON 对象")
    version = _required_string(payload, "schema_version")
    if version != SCHEMA_VERSION:
        raise ValueError(f"暂不支持 schema_version={version}，当前仅支持 {SCHEMA_VERSION}")
    question_id = _required_string(payload, "question_id")
    question_text = _required_string(payload, "question_text")

    raw_topics = payload.get("topic_entities")
    if not isinstance(raw_topics, list) or not raw_topics:
        raise ValueError("topic_entities 必须是非空数组")
    topic_entities = [str(value) for value in raw_topics]

    macro_subgraph = payload.get("macro_subgraph")
    if not isinstance(macro_subgraph, dict):
        raise ValueError("macro_subgraph 必须是 JSON 对象")
    raw_nodes = macro_subgraph.get("nodes")
    if not isinstance(raw_nodes, list):
        raise ValueError("macro_subgraph.nodes 必须是数组")

    nodes: dict[str, dict[str, Any]] = {}
    for item in raw_nodes:
        if not isinstance(item, dict) or not item.get("entity_id"):
            raise ValueError("每个 node 都必须包含 entity_id")
        entity_id = str(item["entity_id"])
        if entity_id in nodes:
            raise ValueError(f"发现重复实体 ID：{entity_id}")
        nodes[entity_id] = item
    missing_topics = [entity_id for entity_id in topic_entities if entity_id not in nodes]
    if missing_topics:
        raise ValueError(f"topic_entities 中的实体未出现在 nodes：{missing_topics}")

    relation_labels = _relation_labels(payload, macro_subgraph)
    raw_triples = macro_subgraph.get("triples")
    if not isinstance(raw_triples, list):
        raise ValueError("macro_subgraph.triples 必须是数组")
    triples: list[Triple] = []
    for index, item in enumerate(raw_triples):
        if isinstance(item, (list, tuple)) and len(item) == 3:
            head_id, relation_id, tail_id = (str(value) for value in item)
        elif isinstance(item, dict):
            try:
                head_id = str(item["head_id"])
                relation_id = str(item["relation_id"])
                tail_id = str(item["tail_id"])
            except KeyError as error:
                raise ValueError("对象形式三元组必须包含 head_id/relation_id/tail_id") from error
        else:
            raise ValueError("每个 triple 必须是 [head_id, relation_id, tail_id]")
        if head_id not in nodes or tail_id not in nodes:
            raise ValueError(f"三元组 {index} 引用了 nodes 中不存在的实体")
        triples.append(
            Triple(
                head=head_id,
                relation=relation_id,
                tail=tail_id,
                id=f"triple_{index}",
                head_label=str(nodes[head_id].get("label") or head_id),
                relation_label=relation_labels.get(relation_id, relation_id),
                tail_label=str(nodes[tail_id].get("label") or tail_id),
            )
        )
    return question_id, question_text, topic_entities, triples, nodes


def build_micro_evidence_subgraph(
    payload: dict[str, Any],
    retriever: MicroRetriever | None = None,
    top_k: int = 20,
    threshold: float = 0.0,
    require_mlp: bool = False,
) -> dict[str, Any]:
    """运行微观检索并生成供 GNN 直接消费的 v0.1 JSON。"""

    if top_k <= 0:
        raise ValueError("top_k 必须大于 0")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold 必须在 0 到 1 之间")
    retriever = retriever or MicroRetriever()
    if require_mlp and retriever.model is None:
        raise ValueError("正式 MLP 模式已启用，但没有加载模型文件")
    question_id, question_text, topics, triples, nodes = parse_macro_subgraph(payload)
    dde = directional_distance_encoding(triples, topics, retriever.rounds)
    # DDE 只计算一次，并同时供三元组评分和 GNN 交接结果复用。
    result = retriever.retrieve(
        question_text,
        topics,
        triples,
        top_k,
        threshold,
        precomputed_dde=dde,
    )

    selected_entities: set[str] = set()
    evidence_triples: list[dict[str, Any]] = []
    for evidence in result.evidence:
        triple = evidence.triple
        selected_entities.update((triple.head, triple.tail))
        evidence_triples.append(
            {
                "triple": [triple.head, triple.relation, triple.tail],
                "relevance_score": round(evidence.score, 6),
                "dde": {
                    "head_dde": [round(value, 6) for value in dde[triple.head]],
                    "tail_dde": [round(value, 6) for value in dde[triple.tail]],
                },
            }
        )

    entity_dde: dict[str, list[float]] = {}
    node_features = []
    for entity_id in sorted(selected_entities):
        node = nodes[entity_id]
        label = str(node.get("label") or entity_id)
        rounded_dde = [round(value, 6) for value in dde[entity_id]]
        entity_dde[entity_id] = rounded_dde
        feature = {
            "entity_id": entity_id,
            "label": label,
            # 原型阶段直接提供真实可用的 128 维向量，不再伪造 emb_Q... 索引。
            "text_embedding": [round(value, 8) for value in retriever.encoder.encode(label)],
        }
        if node.get("text_embedding_id"):
            # 共享向量库上线后，上游提供真实索引；内联向量仍作为兼容回退。
            feature["text_embedding_id"] = str(node["text_embedding_id"])
        node_features.append(feature)

    selected_relation_labels = {
        evidence.triple.relation: evidence.triple.relation_text
        for evidence in result.evidence
    }
    relation_to_id = {
        relation_id: index
        for index, relation_id in enumerate(sorted(selected_relation_labels))
    }
    dde_dim = 2 + 4 * retriever.rounds
    return {
        "schema_version": SCHEMA_VERSION,
        "question_id": question_id,
        "evidence_triples": evidence_triples,
        "top_k": top_k,
        "node_features": node_features,
        "entity_dde": entity_dde,
        "relation_labels": selected_relation_labels,
        "relation_to_id": relation_to_id,
        "feature_spec": {
            "dde_dim": dde_dim,
            "dde_block_dim": 2,
            "dde_rounds": retriever.rounds,
            "dde_order": [
                "topic_one_hot",
                *[f"incoming_round_{index}" for index in range(1, retriever.rounds + 1)],
                *[f"outgoing_round_{index}" for index in range(1, retriever.rounds + 1)],
            ],
            "text_embedding_dim": retriever.encoder.dimensions,
            "text_embedding_source": "inline_hashing",
            "gnn_input_dim": retriever.encoder.dimensions + dde_dim,
        },
        "scoring": {
            "scorer_type": retriever.scorer_type,
            "model_loaded": retriever.model is not None,
            "threshold": threshold,
        },
    }

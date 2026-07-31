# -*- coding: utf-8 -*-
"""
宏观子图裁剪模块

核心创新点：
基于超关系引导的宏观 BFS 裁剪：
由“全图搜索”转变为“高相关领域子图搜索”，在保证候选覆盖率的同时减少后续 GNN 推理噪音。

"""

from typing import List, Dict, Set, Tuple

import config
from kg_store import KGStore


def _initial_importance(entity_id: str,
                         kg: KGStore,
                         is_topic: bool,
                         linking_confidence: float,
                         best_hyper_score: float) -> float:
    """
    节点初始重要性 = 实体链接置信度 / 超关系匹配得分 与 节点中心性的加权组合，
    对应接口文档字段说明："由实体链接置信度、超关系匹配程度及节点中心性综合计算"。
    """
    centrality = kg.normalized_degree(entity_id)
    if is_topic:
        # 主题实体本身就是问题明确提到的实体，链接置信度权重更高
        score = 0.7 * linking_confidence + 0.3 * centrality
    else:
        # 非主题实体的重要性主要来自把它引入子图的那条高分超关系
        score = 0.6 * best_hyper_score + 0.4 * centrality
    return round(min(1.0, max(0.0, score)), 4)


def prune_macro_subgraph(topic_entities: List[str],
                          entity_linking: List[Dict],
                          selected_hyper_relations: List[Dict],
                          kg: KGStore,
                          max_hops: int = config.MAX_HOPS) -> Dict:
    """
    从 topic_entities 出发，仅沿着 selected_hyper_relations 覆盖的关系扩展，
    做最多 max_hops 跳的 BFS，返回裁剪后的宏观子图：
    {"nodes": [...], "triples": [...]}，字段与接口文档一致。
    """
    allowed_hr_ids = {hr["relation_id"] for hr in selected_hyper_relations}
    hr_score_by_id = {hr["relation_id"]: hr["score"] for hr in selected_hyper_relations}

    linking_conf = {e["entity_id"]: e["confidence"] for e in entity_linking}

    visited_nodes: Dict[str, Dict] = {}
    # 记录非主题节点是被哪一条超关系、以多高分数带入子图的，用于计算 initial_importance
    best_incoming_hr_score: Dict[str, float] = {}
    triples: List[Tuple[str, str, str]] = []
    seen_triples: Set[Tuple[str, str, str]] = set()

    frontier = list(topic_entities)
    frontier_seen: Set[str] = set(frontier)

    for entity_id in topic_entities:
        visited_nodes[entity_id] = {"is_topic": True}

    for _hop in range(max_hops):
        next_frontier = []
        for head in frontier:
            for relation_id, tail in kg.neighbors(head):
                hr_id = kg.hyper_relation_of(relation_id)
                if hr_id not in allowed_hr_ids:
                    continue  # 只沿高分超关系约束的方向扩展 —— 这是"宏观检索"的核心裁剪逻辑

                triple = (head, relation_id, tail)
                if triple not in seen_triples:
                    seen_triples.add(triple)
                    triples.append(triple)

                hr_score = hr_score_by_id.get(hr_id, 0.0)
                if tail not in visited_nodes:
                    visited_nodes[tail] = {"is_topic": tail in topic_entities}
                best_incoming_hr_score[tail] = max(best_incoming_hr_score.get(tail, 0.0), hr_score)

                if tail not in frontier_seen:
                    frontier_seen.add(tail)
                    next_frontier.append(tail)
        frontier = next_frontier
        if not frontier:
            break

    nodes = []
    for entity_id, info in visited_nodes.items():
        importance = _initial_importance(
            entity_id=entity_id,
            kg=kg,
            is_topic=info["is_topic"],
            linking_confidence=linking_conf.get(entity_id, 0.0),
            best_hyper_score=best_incoming_hr_score.get(entity_id, 0.0),
        )
        nodes.append({
            "entity_id": entity_id,
            "label": kg.entity_label(entity_id),
            "is_topic": info["is_topic"],
            "initial_importance": importance,
        })
    # 主题实体优先展示，其余按重要性降序，便于人工检查
    nodes.sort(key=lambda n: (not n["is_topic"], -n["initial_importance"]))

    return {
        "nodes": nodes,
        "triples": [list(t) for t in triples],
    }

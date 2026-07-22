"""宏观推理入口：特征 → 评分 → 排序 → 提路径。"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from gnn.macro_gnn.features import build_macro_features
from gnn.scoring.node_scorer import load_and_score
from gnn.scoring.ranker import filter_top_k
from gnn.pathgen.bfs import extract_shortest_path

def run_macro_inference(model, graph_data, topic_entity_ids, entity_embeddings,
                        entity_labels=None, relation_labels=None, relation_id_map=None, top_k=10, max_hops=3, device=None):
    bert = torch.stack([entity_embeddings[eid] for eid in graph_data.node_ids])
    graph_data.node_features = build_macro_features(bert, graph_data.node_ids, topic_entity_ids)
    scores = load_and_score(model, graph_data, device)
    cands = filter_top_k(scores, graph_data.node_ids, topic_entity_ids, top_k)
    if entity_labels:
        for c in cands:
            if c["entity_id"] in entity_labels: c["label"] = entity_labels[c["entity_id"]]
    paths = []
    for c in cands:
        p = extract_shortest_path(graph_data.edge_index, graph_data.edge_type, graph_data.node_ids, graph_data.node_id_to_idx, topic_entity_ids, c["entity_id"], max_hops, relation_id_map, entity_labels, relation_labels)
        if p: paths.append(p)
    return {"candidate_answers": cands, "reasoning_paths": paths}
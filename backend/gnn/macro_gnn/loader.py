"""宏观数据装配：三元组 → GraphData + BERT+is_topic 特征。"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gnn.core.base_loader import triples_to_graph_data
from gnn.macro_gnn.features import build_macro_features
import torch

def assemble_macro_subgraph(triples, entity_embeddings, topic_entity_ids, relation_to_id=None):
    gd = triples_to_graph_data(triples, relation_to_id=relation_to_id)
    bert = torch.stack([entity_embeddings[eid] for eid in gd.node_ids])
    gd.node_features = build_macro_features(bert, gd.node_ids, topic_entity_ids)
    return gd
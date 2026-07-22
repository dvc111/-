"""微观数据装配：证据子图 + DDE → GraphData。"""
import sys; from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gnn.core.base_loader import triples_to_graph_data
from gnn.micro_gnn.features import build_micro_features
import torch

def assemble_micro_subgraph(triples, entity_embeddings, entity_dde, relation_to_id=None):
    gd = triples_to_graph_data(triples, relation_to_id=relation_to_id)
    bert = torch.stack([entity_embeddings[eid] for eid in gd.node_ids])
    dde = torch.stack([entity_dde.get(eid, torch.zeros(64)) for eid in gd.node_ids])
    gd.node_features = build_micro_features(bert, dde)
    return gd
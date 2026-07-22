"""微观特征：BERT + DDE 有向距离编码。"""
import torch
def build_micro_features(bert_embeddings, dde_matrix):
    if dde_matrix is None: return bert_embeddings
    return torch.cat([bert_embeddings, dde_matrix], dim=-1)
"""宏观特征：BERT + is_topic 标志位。"""
import torch
def build_macro_features(bert_embeddings, entity_ids, topic_entity_ids):
    N = bert_embeddings.size(0)
    is_topic = torch.zeros(N, 1)
    topic_set = set(topic_entity_ids)
    for i, eid in enumerate(entity_ids):
        if eid in topic_set: is_topic[i] = 1.0
    return torch.cat([bert_embeddings, is_topic], dim=-1)
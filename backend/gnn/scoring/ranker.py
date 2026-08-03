import torch

def filter_top_k(scores, node_ids, topic_entity_ids=None, top_k=10):
    topic_set = set(topic_entity_ids or [])
    cands = [(i, float(scores[i])) for i in range(scores.size(0)) if node_ids[i] not in topic_set]
    cands.sort(key=lambda x: x[1], reverse=True)
    return [{"entity_id": node_ids[i], "prob": s} for i, s in cands[:top_k]]
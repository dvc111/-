"""GNN introspection for the macro demo (subgraph, fused features, message passing)."""

from __future__ import annotations

import math

import torch


def _rel_label(relation_labels, relation_id):
    if not relation_labels:
        return relation_id
    return relation_labels.get(relation_id, relation_id)


def _cosine(left, right):
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (left_norm * right_norm)))


def run_macro_introspection(model, graph_data, triples, relation_id_map=None,
                            entity_labels=None, relation_labels=None,
                            topic_entity_ids=None, text_dim=None,
                            question_embedding=None, device=None):
    if text_dim is None:
        text_dim = int(graph_data.node_features.size(-1)) - 1
    target_device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model.eval()
    model.to(target_device)
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        raise ValueError("introspection requires an RGCN model with .encoder")
    x = graph_data.node_features.to(target_device)
    edge_index = graph_data.edge_index.to(target_device)
    edge_type = graph_data.edge_type.to(target_device)
    node_ids = graph_data.node_ids

    node_labels = {
        eid: entity_labels[eid] if entity_labels and eid in entity_labels else eid
        for eid in node_ids
    }
    relation_type_of = {}
    if relation_id_map:
        for type_id, rid in relation_id_map.items():
            relation_type_of[rid] = int(type_id)

    neighbor_count = {eid: 0 for eid in node_ids}
    edge_count = {eid: 0 for eid in node_ids}
    neighbor_sets = {eid: set() for eid in node_ids}
    for head, _relation_id, tail in triples:
        if head in edge_count:
            edge_count[head] += 1
        if tail in edge_count:
            edge_count[tail] += 1
        if head in neighbor_sets and tail in neighbor_sets:
            neighbor_sets[head].add(tail)
            neighbor_sets[tail].add(head)
    for eid in node_ids:
        neighbor_count[eid] = len(neighbor_sets[eid])

    qvec = None
    if question_embedding is not None:
        qvec = question_embedding.tolist() if hasattr(question_embedding, "tolist") else list(question_embedding)

    nodes = []
    for i, eid in enumerate(node_ids):
        feat = graph_data.node_features[i].tolist()
        text_vec = feat[:text_dim]
        topic_value = float(feat[text_dim]) if len(feat) > text_dim else 0.0
        norm = math.sqrt(sum(v * v for v in text_vec)) if text_vec else 0.0
        nodes.append({
            "entity_id": eid,
            "label": node_labels[eid],
            "is_topic": bool(topic_value >= 0.5),
            "semantic_score": round(_cosine(qvec, text_vec), 4) if qvec else None,
            "neighbor_count": neighbor_count[eid],
            "edge_count": edge_count[eid],
            "embedding_norm": round(norm, 6),
            "embedding_mean": round(sum(text_vec) / len(text_vec), 6) if text_vec else 0.0,
            "feature_dim": text_dim,
            "input_dim": text_dim + 1,
            "importance": 0.0,
        })

    edges = []
    for head, relation_id, tail in triples:
        edges.append({
            "source": head,
            "target": tail,
            "relation_id": relation_id,
            "label": _rel_label(relation_labels, relation_id),
            "relation_type": relation_type_of.get(relation_id),
        })

    layers = encoder.layers
    layer_results = []
    prev_x = x
    for layer_index, layer in enumerate(layers):
        out, trace = layer.forward_with_trace(x, edge_index, edge_type)
        edge_messages = []
        node_incoming = {eid: {} for eid in node_ids}
        self_loop_norm = {eid: 0.0 for eid in node_ids}
        max_norm = 0.0
        for group in trace["messages"]:
            rel_type = group["relation"]
            rid = relation_id_map.get(rel_type, str(rel_type)) if relation_id_map else str(rel_type)
            src_ids = [node_ids[i] for i in group["src"].tolist()]
            dst_ids = [node_ids[i] for i in group["dst"].tolist()]
            norms = [float(v) for v in group["message"].norm(dim=1).tolist()]
            for src_id, dst_id, norm in zip(src_ids, dst_ids, norms):
                edge_messages.append({
                    "source": src_id,
                    "target": dst_id,
                    "relation_id": rid,
                    "relation_type": rel_type,
                    "norm": round(norm, 6),
                })
                max_norm = max(max_norm, norm)
                bucket = node_incoming[dst_id].setdefault(rid, {"count": 0, "norm": 0.0})
                bucket["count"] += 1
                bucket["norm"] += norm
        if trace["self_loop"] is not None:
            norms = [float(v) for v in trace["self_loop"].norm(dim=1).tolist()]
            for eid, norm in zip(node_ids, norms):
                self_loop_norm[eid] = norm
        if out.size(-1) == prev_x.size(-1):
            delta_norms = [float(v) for v in (out - prev_x).norm(dim=1).tolist()]
        else:
            prev_norms = [float(v) for v in prev_x.norm(dim=1).tolist()]
            out_norms_for_delta = [float(v) for v in out.norm(dim=1).tolist()]
            delta_norms = [abs(o - p) for o, p in zip(out_norms_for_delta, prev_norms)]
        out_norms = [float(v) for v in out.norm(dim=1).tolist()]
        node_states = []
        for i, eid in enumerate(node_ids):
            incoming = [
                {
                    "relation_id": rid,
                    "label": _rel_label(relation_labels, rid),
                    "count": info["count"],
                    "norm": round(info["norm"], 6),
                    "normalized": round(info["norm"] / max_norm, 6) if max_norm else 0.0,
                }
                for rid, info in node_incoming[eid].items()
            ]
            incoming.sort(key=lambda item: item["norm"], reverse=True)
            node_states.append({
                "entity_id": eid,
                "label": node_labels[eid],
                "incoming": incoming,
                "self_loop_norm": round(self_loop_norm[eid], 6),
                "delta": round(delta_norms[i], 6),
                "embedding_norm": round(out_norms[i], 6),
            })
        edge_messages.sort(key=lambda item: item["norm"], reverse=True)
        for edge_msg in edge_messages:
            edge_msg["normalized"] = round(edge_msg["norm"] / max_norm, 6) if max_norm else 0.0
        layer_results.append({
            "layer_index": layer_index,
            "max_message_norm": round(max_norm, 6),
            "edge_messages": edge_messages,
            "node_states": node_states,
        })
        prev_x = out
        x = out

    node_scores = []
    classifier = getattr(model, "classifier", None)
    if classifier is not None:
        with torch.no_grad():
            probs = torch.softmax(classifier(x), dim=-1)[:, 1].tolist()
        node_scores = [
            {"entity_id": eid, "prob": round(float(prob), 6)}
            for eid, prob in zip(node_ids, probs)
        ]
    importance_map = {item["entity_id"]: item["prob"] for item in node_scores}
    for node in nodes:
        node["importance"] = importance_map.get(node["entity_id"], 0.0)

    return {
        "feature_spec": {
            "text_dim": text_dim,
            "is_topic_dim": 1,
            "input_dim": text_dim + 1,
            "encoder": "bert" if text_dim == 768 else "hashing",
        },
        "nodes": nodes,
        "edges": edges,
        "node_scores": node_scores,
        "layers": layer_results,
    }

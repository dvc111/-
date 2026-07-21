"""双向 BFS 最短路径提取。

给定子图、主题实体 ID 列表和候选答案 ID，
通过双向广度优先搜索找出从任意主题实体到候选答案的最短路径，
限制最大跳数，多条等长路径时按关系语义相似度选最优。
宏观和微观的 inference.py 都会调用此模块。"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from collections import deque
from typing import Optional


def _build_adj(edge_index, edge_type, num_nodes):
    fwd = [[] for _ in range(num_nodes)]
    bwd = [[] for _ in range(num_nodes)]
    for i in range(edge_index.size(1)):
        s, d, r = int(edge_index[0, i]), int(edge_index[1, i]), int(edge_type[i])
        fwd[s].append((d, r))
        bwd[d].append((s, r))
    return fwd, bwd


def _reconstruct(mid, fwd_t, bwd_t, src):
    seg = []
    cur = mid
    while cur not in src:
        cur = fwd_t[cur][0]
    start = cur
    cur = mid
    while cur != start:
        p, r = fwd_t[cur]
        seg.append(cur); seg.append(r)
        cur = p
    seg.append(cur)
    seg.reverse()
    cur = mid
    while cur in bwd_t:
        n, r = bwd_t[cur]
        if n == -1:
            break
        seg.append(r); seg.append(n)
        cur = n
    return seg


def _to_path(indices, node_ids, rmap, elbl, rlbl):
    out = []
    for i, v in enumerate(indices):
        if i % 2 == 0:
            eid = node_ids[v]
            d = {"entity_id": eid}
            if elbl and eid in elbl:
                d["label"] = elbl[eid]
            out.append(d)
        else:
            rid = rmap.get(v, f"R{v:03d}")
            d = {"relation_id": rid}
            if rlbl and rid in rlbl:
                d["label"] = rlbl[rid]
            out.append(d)
    return out


def _sim(path, rembs, qemb):
    rels = path[1::2]
    vecs = [rembs[r] for r in rels if r in rembs]
    if not vecs:
        return 0.0
    mean_emb = torch.stack(vecs).mean(dim=0, keepdim=True)
    return float(F.cosine_similarity(mean_emb, qemb.unsqueeze(0)).item())


def extract_shortest_path(
    edge_index, edge_type, node_ids, node_id_to_idx,
    topic_entity_ids, answer_entity_id,
    max_hops=3,
    entity_labels=None, relation_labels=None,
    relation_embeddings=None, question_embedding=None,
    relation_id_map=None,
):
    """双向 BFS 提取从主题实体到候选答案的最短路径。"""
    aidx = node_id_to_idx.get(answer_entity_id)
    if aidx is None:
        return None
    tidxs = [node_id_to_idx[e] for e in topic_entity_ids if e in node_id_to_idx]
    if not tidxs:
        return None

    fwd_a, bwd_a = _build_adj(edge_index, edge_type, len(node_ids))

    fwd_v, fwd_d, fwd_q, src = {}, {}, deque(), {}
    for t in tidxs:
        fwd_v[t] = (-1, -1); fwd_d[t] = 0; fwd_q.append(t); src[t] = t

    bwd_v, bwd_d, bwd_q = {}, {}, deque()
    bwd_v[aidx] = (-1, -1); bwd_d[aidx] = 0; bwd_q.append(aidx)

    best = max_hops + 1
    paths = []

    def _layer(q, vis, dist, adj, is_fwd):
        nonlocal best
        hits = []
        for _ in range(len(q)):
            u = q.popleft()
            d = dist[u] + 1
            if d >= best:
                continue
            other_v = bwd_v if is_fwd else fwd_v
            other_d = bwd_d if is_fwd else fwd_d
            for v, r in adj[u]:
                if v in vis:
                    continue
                vis[v] = (u, r)
                dist[v] = d
                if v in other_v:
                    tot = d + other_d[v]
                    if tot <= max_hops and tot <= best:
                        best = tot
                        hits.append(v)
                q.append(v)
        return hits

    while fwd_q and bwd_q:
        hf = _layer(fwd_q, fwd_v, fwd_d, fwd_a, True)
        hb = _layer(bwd_q, bwd_v, bwd_d, bwd_a, False)
        for hits in [hf, hb]:
            for node in hits:
                if fwd_d[node] + bwd_d[node] == best:
                    p = _reconstruct(node, fwd_v, bwd_v, src)
                    if p not in paths:
                        paths.append(p)
        if paths:
            break

    if not paths:
        return None

    score = 0.0
    if len(paths) > 1 and relation_embeddings and question_embedding is not None:
        best_p = max(paths, key=lambda p: _sim(p, relation_embeddings, question_embedding))
        score = _sim(best_p, relation_embeddings, question_embedding)
    else:
        best_p = paths[0]

    rmap = relation_id_map or {}
    if not rmap:
        mr = int(edge_type.max().item()) if edge_type.numel() > 0 else 0
        rmap = {i: f"R{i:03d}" for i in range(mr + 1)}

    return {
        "answer_entity_id": answer_entity_id,
        "path": _to_path(best_p, node_ids, rmap, entity_labels, relation_labels),
        "path_score": round(score, 4),
    }

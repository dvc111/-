"""双向 BFS 最短路径提取：从主题实体到候选答案，前后夹击搜索。"""

from collections import deque


def extract_shortest_path(edge_index, edge_type, node_ids, node_id_to_idx,
                          topic_entity_ids, answer_entity_id, max_hops=3,
                          relation_id_map=None, entity_labels=None, relation_labels=None):
    aidx = node_id_to_idx.get(answer_entity_id)
    if aidx is None: return None
    tidxs = [node_id_to_idx[e] for e in topic_entity_ids if e in node_id_to_idx]
    if not tidxs: return None
    N = len(node_ids)

    # 建邻接表
    fwd_a = [[] for _ in range(N)]; bwd_a = [[] for _ in range(N)]
    for i in range(edge_index.size(1)):
        s, d, r = int(edge_index[0, i]), int(edge_index[1, i]), int(edge_type[i])
        fwd_a[s].append((d, r)); bwd_a[d].append((s, r))

    # 正向 BFS（主题实体 → 外）
    fwd_v, fwd_d, fwd_q, src = {}, {}, deque(), {}
    for t in tidxs: fwd_v[t] = (-1, -1); fwd_d[t] = 0; fwd_q.append(t); src[t] = t

    # 反向 BFS（候选答案 → 外）
    bwd_v, bwd_d, bwd_q = {}, {}, deque()
    bwd_v[aidx] = (-1, -1); bwd_d[aidx] = 0; bwd_q.append(aidx)

    best = max_hops + 1
    while fwd_q and bwd_q:
        for _ in range(len(fwd_q)):
            u = fwd_q.popleft(); d = fwd_d[u] + 1
            if d >= best: continue
            for v, r in fwd_a[u]:
                if v not in fwd_v:
                    fwd_v[v] = (u, r); fwd_d[v] = d; fwd_q.append(v)
                    if v in bwd_v: best = d + bwd_d[v]
        if best <= max_hops: break

        for _ in range(len(bwd_q)):
            u = bwd_q.popleft(); d = bwd_d[u] + 1
            if d >= best: continue
            for v, r in bwd_a[u]:
                if v not in bwd_v:
                    bwd_v[v] = (u, r); bwd_d[v] = d; bwd_q.append(v)
                    if v in fwd_v: best = d + fwd_d[v]
        if best <= max_hops: break

    if best > max_hops: return None

    # 找相遇点（离主题实体最近的那个）
    mid = None
    for v in fwd_v:
        if v in bwd_v and fwd_d[v] + bwd_d[v] == best:
            if mid is None or fwd_d[v] < fwd_d[mid]: mid = v
    if mid is None: return None

    # 回溯拼接路径
    seg = []; cur = mid
    while cur not in src:
        p, r = fwd_v[cur]; seg.append(cur); seg.append(r); cur = p
    seg.append(cur); seg.reverse()
    cur = mid
    while cur in bwd_v:
        n, r = bwd_v[cur]
        if n == -1: break
        seg.append(r); seg.append(n); cur = n

    # 格式化输出
    mr = int(edge_type.max().item()) if edge_type.numel() > 0 else 0
    rmap = relation_id_map or {i: f"R{i}" for i in range(mr + 1)}
    path = []
    for i, v in enumerate(seg):
        if i % 2 == 0:
            eid = node_ids[v]; d = {"entity_id": eid}
            if entity_labels and eid in entity_labels: d["label"] = entity_labels[eid]
            path.append(d)
        else:
            rid = rmap.get(v, str(v)); d = {"relation_id": rid}
            if relation_labels and rid in relation_labels: d["label"] = relation_labels[rid]
            path.append(d)
    return {"answer_entity_id": answer_entity_id, "path": path, "path_score": 0.0}
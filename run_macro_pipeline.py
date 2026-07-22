"""宏观检索 -> GNN 推理，全流程演示（宏观+GNN 软著对应的验证脚本）。"""

import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import torch
from gnn.core.model import RGCNNodeClassifier
from gnn.macro_gnn.inference import run_macro_inference


def main():
    print("=" * 50)
    print("宏观 + GNN 全流程演示")
    print("=" * 50)

    # ── 1. 加载宏观子图数据 ──
    macro_path = os.path.join("backend", "retrieval", "examples", "macro_subgraph_v0.1.json")
    with open(macro_path, encoding="utf-8") as f:
        payload = json.load(f)

    print(f"\n提问: {payload['question_text']}")
    print(f"主题实体: {payload['topic_entities']}")
    topic_ids = payload["topic_entities"]
    macro_sg = payload["macro_subgraph"]
    triples = [tuple(t) for t in macro_sg["triples"]]
    rel_labels = payload["relation_labels"]

    # 造 BERT 嵌入（768维，演示用随机数）
    entity_embeddings = {}
    entity_labels = {}
    for node in macro_sg["nodes"]:
        eid = node["entity_id"]
        entity_embeddings[eid] = torch.randn(768)
        entity_labels[eid] = node["label"]

    print(f"\n宏观子图: {len(macro_sg['nodes'])} 实体, {len(triples)} 三元组")

    # ── 2. 装盘 → GraphData ──
    from gnn.macro_gnn.loader import assemble_macro_subgraph
    gd = assemble_macro_subgraph(triples, entity_embeddings, topic_ids)
    print(f"GraphData: {gd.num_nodes} 节点, {gd.num_edges} 边, 特征 {list(gd.node_features.shape)}")
    is_topic_col = gd.node_features[:, -1].tolist()
    print(f"is_topic 列: {is_topic_col}")

    # ── 3. GNN 推理 ──
    num_rels = gd.num_relations
    model = RGCNNodeClassifier(in_dim=769, hidden_dim=64, num_relations=num_rels)
    relation_id_map = {i: rid for i, rid in enumerate(sorted(rel_labels.keys()))}

    result = run_macro_inference(
        model=model,
        graph_data=gd,
        topic_entity_ids=topic_ids,
        entity_embeddings=entity_embeddings,
        entity_labels=entity_labels,
        relation_labels=rel_labels,
        relation_id_map=relation_id_map,
        top_k=5,
        max_hops=3,
    )

    print(f"\nGNN 推理结果:")
    for c in result["candidate_answers"]:
        label = c.get("label", c["entity_id"])
        print(f"  候选: {label} ({c['prob']:.3f})")

    from gnn.pathgen.verbalizer import verbalize
    for rp in result["reasoning_paths"]:
        ans_id = rp["answer_entity_id"]
        ans_label = entity_labels.get(ans_id, ans_id)
        print(f"  路径 {ans_label}: {verbalize(rp['path'])}")

    print("\n" + "=" * 50)
    print("宏观+GNN 演示完成")
    print("=" * 50)


if __name__ == "__main__":
    main()
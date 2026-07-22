""" 微观检索 -> GNN 推理，全流程演示。"""

import sys, json, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import torch
from retrieval.micro_rag import build_micro_evidence_subgraph, MicroRetriever
from gnn.core.model import RGCNNodeScorer
from gnn.micro_gnn.inference import run_micro_inference

def main():
    print("=" * 50)
    print("微观 + GNN")
    print("=" * 50)

    # 1. 加载示例宏观子图
    macro_path = os.path.join("backend", "retrieval", "examples", "macro_subgraph_v0.1.json")
    with open(macro_path, encoding="utf-8") as f:
        payload = json.load(f)
    print(f"\n提问: {payload['question_text']}")
    print(f"主题实体: {payload['topic_entities']}")

    # 2. 微观检索（DDE + MLP 评分）
    retriever = MicroRetriever()
    evidence = build_micro_evidence_subgraph(payload, retriever, top_k=20)
    print(f"\n微观检索: 筛出 {len(evidence['evidence_triples'])} 条证据三元组")
    print(f"  DDE 维度: {evidence['feature_spec']['dde_dim']}")
    print(f"  GNN 输入维度: {evidence['feature_spec']['gnn_input_dim']}")

    ets = evidence["evidence_triples"]
    entity_dde = evidence["entity_dde"]
    node_feats = evidence["node_features"]
    rel_labels = evidence["relation_labels"]

    # 3. 组装 GNN 输入
    triples = [et["triple"] for et in ets]
    entity_embeddings = {}
    entity_labels = {}
    for nf in node_feats:
        entity_embeddings[nf["entity_id"]] = torch.tensor(nf["text_embedding"])
        entity_labels[nf["entity_id"]] = nf["label"]

    dde = {}
    for eid, vec in entity_dde.items():
        dde[eid] = torch.tensor(vec)

    from gnn.micro_gnn.loader import assemble_micro_subgraph
    gd = assemble_micro_subgraph(triples, entity_embeddings, dde)
    print(f"\n子图: {gd.num_nodes} 节点, {gd.num_edges} 边")

    # 4. GNN 推理
    in_dim = evidence["feature_spec"]["gnn_input_dim"]
    num_rels = gd.num_relations
    model = RGCNNodeScorer(in_dim=in_dim, hidden_dim=64, num_relations=num_rels)
    relation_id_map = {i: rid for i, rid in enumerate(sorted(rel_labels))} if rel_labels else None

    topic_ids = payload["topic_entities"]
    result = run_micro_inference(
        model=model,
        graph_data=gd,
        topic_entity_ids=topic_ids,
        entity_embeddings=entity_embeddings,
        entity_dde=dde,
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
    print("演示完成")
    print("=" * 50)

if __name__ == "__main__":
    main()
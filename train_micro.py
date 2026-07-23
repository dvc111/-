"""训练微观 GNN 模型。用 3 个示例自动生成训练数据，保存 micro_model.pth。"""

import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))

import torch
from retrieval.micro_rag import build_micro_evidence_subgraph, MicroRetriever
from gnn.core.model import RGCNNodeScorer
from gnn.micro_gnn.trainer import train_micro_model
from gnn.micro_gnn.loader import assemble_micro_subgraph


def main():
    train_configs = [
        {"json": "backend/retrieval/examples/macro_subgraph_v0.1.json",
         "answers": ["Q2031", "Q2099"]},
        {"json": "backend/retrieval/examples/einstein_macro_subgraph_v0.1.json",
         "answers": ["Q_GERMAN"]},
        {"json": "backend/retrieval/examples/book_macro_subgraph_v0.1.json",
         "answers": ["Q_BEIJING"]},
    ]

    retriever = MicroRetriever()
    train_data = []
    in_dim = None
    max_rels = 0

    for cfg in train_configs:
        with open(cfg["json"], encoding="utf-8") as f:
            payload = json.load(f)
        evidence = build_micro_evidence_subgraph(payload, retriever, top_k=20)
        triples = [et["triple"] for et in evidence["evidence_triples"]]
        entity_emb = {nf["entity_id"]: torch.tensor(nf["text_embedding"]) for nf in evidence["node_features"]}
        entity_dde = {eid: torch.tensor(vec) for eid, vec in evidence["entity_dde"].items()}
        gd = assemble_micro_subgraph(triples, entity_emb, entity_dde)

        labels = torch.zeros(gd.num_nodes)
        answer_set = set(cfg["answers"])
        for i, eid in enumerate(gd.node_ids):
            if eid in answer_set:
                labels[i] = 1.0

        train_data.append((gd, labels))
        if in_dim is None:
            in_dim = evidence["feature_spec"]["gnn_input_dim"]
        max_rels = max(max_rels, gd.num_relations)
        print(f"  {payload['question_text']}: {gd.num_nodes} 节点, {gd.num_relations} 种关系")

    model = RGCNNodeScorer(in_dim=in_dim, hidden_dim=64, num_relations=max_rels)
    print(f"\n模型: in_dim={in_dim}, num_relations={max_rels}")
    train_micro_model(model, train_data, epochs=30, save_path="micro_model.pth")
    torch.save(model.state_dict(), "micro_model.pth")
    torch.save(model, "micro_model_full.pth")  # ????????

    # 验证
    for i, cfg in enumerate(train_configs):
        with open(cfg["json"], encoding="utf-8") as f:
            payload = json.load(f)
        evidence = build_micro_evidence_subgraph(payload, retriever, top_k=20)
        triples = [et["triple"] for et in evidence["evidence_triples"]]
        entity_emb = {nf["entity_id"]: torch.tensor(nf["text_embedding"]) for nf in evidence["node_features"]}
        entity_dde = {eid: torch.tensor(vec) for eid, vec in evidence["entity_dde"].items()}
        gd = assemble_micro_subgraph(triples, entity_emb, entity_dde)
        from gnn.micro_gnn.inference import run_micro_inference
        result = run_micro_inference(model=model, graph_data=gd, topic_entity_ids=payload["topic_entities"],
            entity_embeddings=entity_emb, entity_dde=entity_dde,
            entity_labels={nf["entity_id"]: nf["label"] for nf in evidence["node_features"]},
            relation_labels=evidence["relation_labels"],
            relation_id_map={i: rid for i, rid in enumerate(sorted(evidence["relation_labels"].keys()))} if evidence["relation_labels"] else None,
            top_k=5, max_hops=3)
        top = result["candidate_answers"][0]
        print(f"  验证 {payload['question_text']}: 最优候选={top.get('label', top['entity_id'])} ({top['prob']:.3f})")

    print(f"\n模型已保存到 micro_model.pth")


if __name__ == "__main__":
    main()
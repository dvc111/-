"""训练宏观 GNN 模型。用医疗 KG 自动生成训练数据，保存 macro_model.pth。"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "REKNOS_macro"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "REKNOS_macro", "macro_retrieval"))

import torch
from kg_store import KGStore
from pipeline import macro_retrieval
from gnn.core.model import RGCNNodeClassifier
from gnn.macro_gnn.trainer import train_macro_model
from gnn.macro_gnn.loader import assemble_macro_subgraph
from retrieval.micro_rag.text_encoder import HashingTextEncoder


def mock_llm(prompt):
    if "候选超关系" in prompt:
        if "副作用" in prompt or "导致" in prompt:
            return "{不良反应 (Score: 0.90)}: effects\n{禁忌症 (Score: 0.70)}: caution\n"
        if "类别" in prompt or "什么类" in prompt:
            return "{药物分类 (Score: 0.92)}: category\n{适应症 (Score: 0.41)}: indication\n"
        if "华法林" in prompt:
            return "{药物相互作用 (Score: 0.90)}: interaction\n{不良反应 (Score: 0.60)}: effects\n"
        return "{药物相互作用 (Score: 0.92)}: risk\n{禁忌症 (Score: 0.81)}: risk\n"
    if "华法林和什么药有相互作用" in prompt: return "华法林"
    if "布洛芬会导致什么" in prompt: return "布洛芬"
    if "阿司匹林有什么副作用" in prompt: return "阿司匹林"
    if "阿司匹林属于什么类别的药" in prompt: return "阿司匹林"
    if prompt.count("阿司匹林和布洛芬同时服用会有什么风险") >= 2: return "阿司匹林\n布洛芬"
    if "对乙酰氨基酚" in prompt: return "对乙酰氨基酚"
    return ""


def main():
    kg = KGStore(os.path.join(os.path.dirname(__file__), "backend", "retrieval", "REKNOS_macro", "kg", "toy_medical_kg_new1.json"))
    encoder = HashingTextEncoder(128)

    train_configs = [
        ("阿司匹林和布洛芬同时服用会有什么风险？", ["Q2031", "Q2099"]),
        ("布洛芬会导致什么？", ["Q2031", "Q2099"]),
        ("阿司匹林有什么副作用？", ["Q2031", "Q2200", "Q2300"]),
        ("阿司匹林属于什么类别的药？", ["Q3000", "Q3001"]),
        ("华法林和什么药有相互作用？", ["Q1024", "Q1088"]),
    ]

    train_data = []
    in_dim = None

    for question, answers in train_configs:
        result = macro_retrieval(
            question_id="q1", question_text=question, kg=kg,
            llm_fn=mock_llm, top_k_hyper_relations=2, max_hops=2,
        )
        sub = result["macro_subgraph"]
        topic_ids = result["topic_entities"]
        triples = [tuple(t) for t in sub["triples"]]

        entity_embeddings = {}
        entity_labels = {}
        for node in sub["nodes"]:
            eid = node["entity_id"]
            entity_labels[eid] = node["label"]
            entity_embeddings[eid] = torch.tensor(encoder.encode(node["label"]))

        gd = assemble_macro_subgraph(triples, entity_embeddings, topic_ids)

        labels = torch.zeros(gd.num_nodes, dtype=torch.long)
        answer_set = set(answers)
        for i, eid in enumerate(gd.node_ids):
            if eid in answer_set:
                labels[i] = 1

        train_data.append((gd, labels))
        if in_dim is None:
            in_dim = gd.node_features.size(-1)

    max_rels = max(gd.num_relations for gd, _ in train_data)
    model = RGCNNodeClassifier(in_dim=in_dim, hidden_dim=64, num_relations=max_rels)

    print(f"训练: in_dim={in_dim}, num_relations={max_rels}, 数据量={len(train_data)}")
    train_macro_model(model, train_data, epochs=15, save_path="macro_model.pth")
    torch.save(model.state_dict(), "macro_model.pth")
    torch.save(model, "macro_model_full.pth")
    print("已完成: macro_model.pth + macro_model_full.pth")


if __name__ == "__main__":
    main()

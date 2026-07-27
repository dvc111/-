
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "REKNOS_macro"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "REKNOS_macro", "macro_retrieval"))

import torch
from kg_store import KGStore
from pipeline import macro_retrieval
from gnn.core.model import RGCNNodeClassifier
from gnn.macro_gnn.inference import run_macro_inference
from gnn.macro_gnn.loader import assemble_macro_subgraph
from gnn.core.encoder import BertTextEncoder

def mock_llm(prompt):
    if "候选超关系" in prompt:
        if "华法林" in prompt:
            return "{药物相互作用 (Score: 0.90)}: interaction\n{药物分类 (Score: 0.70)}: category\n"
        if "导致" in prompt:
            return "{不良反应 (Score: 0.90)}: effects\n{药物相互作用 (Score: 0.50)}: interaction\n"
        if "副作用" in prompt:
            return "{不良反应 (Score: 0.90)}: effects\n{适应症 (Score: 0.30)}: indication\n"
        if "类别" in prompt or "什么类" in prompt:
            return "{药物分类 (Score: 0.90)}: category\n{药理机制 (Score: 0.50)}: mechanism\n"
        return "{药物相互作用 (Score: 0.95)}: risk\n{禁忌症 (Score: 0.85)}: risk\n"
    if "华法林和什么药有相互作用" in prompt: return "华法林"
    if "布洛芬会导致什么" in prompt: return "布洛芬"
    if "阿司匹林有什么副作用" in prompt: return "阿司匹林"
    if "阿司匹林属于什么类别的药" in prompt: return "阿司匹林"
    if prompt.count("阿司匹林和布洛芬同时服用会有什么风险") >= 2: return "阿司匹林\n布洛芬"
    if "对乙酰氨基酚" in prompt: return "对乙酰氨基酚"
    return ""


def main():
    kg = KGStore(os.path.join(os.path.dirname(__file__), "backend", "retrieval", "REKNOS_macro", "kg", "toy_medical_kg_new1.json"))
    encoder = BertTextEncoder()
    qs = ["阿司匹林和布洛芬同时服用会有什么风险？","阿司匹林有什么副作用？","布洛芬会导致什么？","阿司匹林属于什么类别的药？","华法林和什么药有相互作用？"]
    qi = 0
    if len(sys.argv) > 1:
        try: qi = max(0, min(len(qs)-1, int(sys.argv[1])-1))
        except: pass
    print(f"问题 {qi+1}/{len(qs)}: {qs[qi]}")

    result = macro_retrieval(question_id="q1", question_text=qs[qi], kg=kg, llm_fn=mock_llm, top_k_hyper_relations=2, max_hops=2)
    hrs = result["selected_hyper_relations"]
    print("选中超关系：" + "，".join([f"{hr['label']} ({hr['score']})" for hr in hrs]))
    print(f"宏观子图规模：{len(result['macro_subgraph']['nodes'])}节点，{len(result['macro_subgraph']['triples'])}边")
    print("主题实体：" + " ".join([kg.entity_label(t) for t in result["topic_entities"]]))

    sub = result["macro_subgraph"]; topic_ids = result["topic_entities"]
    triples = [tuple(t) for t in sub["triples"]];
    emb,lbl = {},{}
    for n in sub["nodes"]:
        lbl[n["entity_id"]] = n["label"]; emb[n["entity_id"]] = torch.tensor(encoder.encode(n["label"]))
    ri = sorted({r for _,r,_ in triples}); rm = {i:rid for i,rid in enumerate(ri)}
    rl = {rid:kg.relations[rid]["label"] for rid in ri}
    gd = assemble_macro_subgraph(triples, emb, topic_ids)

    
    mp = os.path.join(os.path.dirname(__file__), "macro_model_full.pth")
    if os.path.exists(mp):
        model = torch.load(mp, map_location="cpu", weights_only=False); print("已加载 macro_model_full.pth")
    else:
        model = RGCNNodeClassifier(in_dim=gd.node_features.size(-1), hidden_dim=64, num_relations=gd.num_relations); print("使用随机权重")

    result = run_macro_inference(model=model, graph_data=gd, topic_entity_ids=topic_ids, entity_embeddings=emb, entity_labels=lbl, relation_labels=rl, relation_id_map=rm, top_k=5, max_hops=3)

    from gnn.pathgen.verbalizer import verbalize
    print("\n候选答案:")
    for c in result["candidate_answers"]: print(f"  {c.get('label', c['entity_id'])} ({c['prob']:.3f})")
    for rp in result["reasoning_paths"]: print(f"  路径: {verbalize(rp['path'])}")
if __name__ == "__main__": main()
"""Macro + GNN server."""
import sys, os, json, mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
BACKEND = str(ROOT / "backend")
MACRO = str(ROOT / "REKNOS_macro")
RET = str(ROOT / "REKNOS_macro" / "macro_retrieval")
for p in [BACKEND, MACRO, RET]:
    if p not in sys.path: sys.path.insert(0, p)
import torch
from kg_store import KGStore
from pipeline import macro_retrieval
from gnn.core.model import RGCNNodeClassifier
from gnn.macro_gnn.inference import run_macro_inference
from gnn.macro_gnn.loader import assemble_macro_subgraph
from retrieval.micro_rag.text_encoder import HashingTextEncoder
FRONTEND = ROOT / "frontend"
KG_PATH = str(ROOT / "REKNOS_macro" / "kg" / "toy_medical_kg.json")
MODEL_PATH = ROOT / "macro_model_full.pth"

def mock_llm(p):
    if "候选超关系" in p:
        return "{药物相互作用 (Score: 0.92)}: risk\n{禁忌症 (Score: 0.81)}: risk\n"
    return "阿司匹林\n布洛芬"

class H(BaseHTTPRequestHandler):
    kg = KGStore(KG_PATH)
    enc = HashingTextEncoder(128)
    d = None

    @classmethod
    def load(cls):
        if cls.d: return
        r = macro_retrieval("q1","阿司匹林和布洛芬同时服用会有什么风险？",cls.kg,llm_fn=mock_llm,top_k_hyper_relations=2,max_hops=2)
        s = r["macro_subgraph"]; t = [tuple(x) for x in s["triples"]]; e,l = {},{}
        _kr = json.load(open(KG_PATH, encoding="utf-8"))
        _rl = _kr.get("relations", {})
        ri = sorted({x for _,x,_ in t})
        rm = {i:rid for i,rid in enumerate(ri)}
        rl = {rid: _rl[rid]["label"] for rid in ri}
        for n in s["nodes"]:
            l[n["entity_id"]] = n["label"]
            e[n["entity_id"]] = torch.tensor(cls.enc.encode(n["label"]))
        g = assemble_macro_subgraph(t, e, r["topic_entities"])
        if MODEL_PATH.exists():
            m = torch.load(str(MODEL_PATH), "cpu", weights_only=False)
        else:
            m = RGCNNodeClassifier(in_dim=g.node_features.size(-1), hidden_dim=64, num_relations=g.num_relations)
        cls.d = {"r":r,"g":g,"e":e,"l":l,"m":m,"tn":[cls.kg.entity_label(t) for t in r["topic_entities"]],"hr":r["selected_hyper_relations"],"s":s,"rl":rl,"rm":rm}

    def do_GET(self):
        if self.path == "/api/macro-infer":
            self.load(); d = self.d
            o = run_macro_inference(model=d["m"],graph_data=d["g"],topic_entity_ids=d["r"]["topic_entities"],entity_embeddings=d["e"],entity_labels=d["l"],relation_labels=d["rl"],relation_id_map=d["rm"],top_k=5,max_hops=3)
            self.send_json({"question":d["r"]["question_text"],"topic_entities":d["tn"],"hyper_relations":d["hr"],"subgraph_size":{"nodes":len(d["s"]["nodes"]),"edges":len(d["s"]["triples"])},"candidates":o["candidate_answers"],"paths":o["reasoning_paths"]})
            return
        if self.path in("/","/macro.html"): self.path = "/macro.html"
        fp = FRONTEND / self.path.lstrip("/")
        if fp.is_file():
            mt,_ = mimetypes.guess_type(str(fp)); self.send_response(200); self.send_header("Content-Type",mt or "application/octet-stream"); self.end_headers()
            with open(fp,"rb") as f: self.wfile.write(f.read())
        else: self.send_response(404); self.end_headers(); self.wfile.write(b"404")

    def send_json(self,d):
        self.send_response(200); self.send_header("Content-Type","application/json; charset=utf-8"); self.end_headers()
        self.wfile.write(json.dumps(d,ensure_ascii=False).encode("utf-8"))

    def log_message(self,f,*a): print(f"[macro] {a[0]} {a[1]}")

if __name__ == "__main__":
    print("http://localhost:8001"); HTTPServer(("0.0.0.0",8001),H).serve_forever()

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
from gnn.core.encoder import BertTextEncoder
FRONTEND = ROOT / "frontend"
KG_PATH = str(ROOT / "backend" / "retrieval" / "REKNOS_macro" / "kg" / "toy_medical_kg_new1.json")
MODEL_PATH = ROOT / "macro_model_full.pth"

QS = [
    "阿司匹林和布洛芬同时服用会有什么风险？",
    "阿司匹林有什么副作用？",
    "布洛芬会导致什么？",
    "阿司匹林属于什么类别的药？",
    "华法林和什么药有相互作用？",
]

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
class H(BaseHTTPRequestHandler):
    kg = KGStore(KG_PATH)
    enc = BertTextEncoder()
    _model = None
    _model_loaded = False

    @classmethod
    def get_model(cls):
        if not cls._model_loaded:
            if MODEL_PATH.exists():
                cls._model = torch.load(str(MODEL_PATH), "cpu", weights_only=False)
            cls._model_loaded = True
        return cls._model

    def run_q(self, qi):
        qtxt = QS[qi]
        r = macro_retrieval("q" + str(qi+1), qtxt, self.kg, llm_fn=mock_llm, top_k_hyper_relations=2, max_hops=2)
        s = r["macro_subgraph"]; t = [tuple(x) for x in s["triples"]]; e,l = {},{}
        _kr = json.load(open(KG_PATH, encoding="utf-8"))
        _rl = _kr.get("relations", {})
        ri = sorted({x for _,x,_ in t})
        rm = {i:rid for i,rid in enumerate(ri)}
        rl = {rid: _rl[rid]["label"] for rid in ri}
        for n in s["nodes"]:
            l[n["entity_id"]] = n["label"]
            e[n["entity_id"]] = torch.tensor(self.enc.encode(n["label"]))
        g = assemble_macro_subgraph(t, e, r["topic_entities"])
        m = self.get_model()
        if m is None:
            m = RGCNNodeClassifier(in_dim=g.node_features.size(-1), hidden_dim=64, num_relations=g.num_relations)
        return {"r":r,"g":g,"e":e,"l":l,"m":m,"tn":[self.kg.entity_label(t) for t in r["topic_entities"]],"hr":r["selected_hyper_relations"],"s":s,"rl":rl,"rm":rm}

    def do_GET(self):
        # parse query string from path
        raw_path = self.path
        path_only = raw_path.split("?")[0]
        qs = raw_path.split("?")[1] if "?" in raw_path else ""
        qp = {}
        if qs:
            for pair in qs.split("&"):
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    qp[k] = v

        if path_only == "/api/macro-infer":
            qi = 0
            if "q" in qp:
                try: qi = max(0, min(4, int(qp["q"]) - 1))
                except: pass
            d = self.run_q(qi)
            o = run_macro_inference(model=d["m"],graph_data=d["g"],topic_entity_ids=d["r"]["topic_entities"],entity_embeddings=d["e"],entity_labels=d["l"],relation_labels=d["rl"],relation_id_map=d["rm"],top_k=5,max_hops=3)
            self.send_json({"question":d["r"]["question_text"],"topic_entities":d["tn"],"hyper_relations":d["hr"],"subgraph_size":{"nodes":len(d["s"]["nodes"]),"edges":len(d["s"]["triples"])},"candidates":o["candidate_answers"],"paths":o["reasoning_paths"]})
            return
        if path_only in("/","/macro.html"):
            self.path = "/macro.html"
            if "q" in qp:
                self.path = "/macro.html?q=" + qp["q"]
        fp = FRONTEND / self.path.lstrip("/").split("?")[0]
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

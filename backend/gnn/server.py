"""微观+GNN 演示服务器（纯内置模块，无需安装）"""

import sys, os, json, mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse, parse_qs

_BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _BACKEND)

import torch
from retrieval.micro_rag import build_micro_evidence_subgraph, MicroRetriever
from gnn.core.model import RGCNNodeScorer
from gnn.micro_gnn.inference import run_micro_inference
from gnn.micro_gnn.loader import assemble_micro_subgraph

EXAMPLES_DIR = Path(__file__).resolve().parents[2] / "backend" / "retrieval" / "examples"
MODEL_PATH = Path(__file__).resolve().parents[2] / "micro_model_full.pth"
FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"

retriever = MicroRetriever()
_cache = {}

def _load(name):
    fname = {"aspirin":"macro_subgraph_v0.1.json","einstein":"einstein_macro_subgraph_v0.1.json","book":"book_macro_subgraph_v0.1.json"}.get(name)
    if not fname: return _load("aspirin")
    with open(EXAMPLES_DIR / fname, encoding="utf-8") as f:
        p = json.load(f)
    e = build_micro_evidence_subgraph(p, retriever, top_k=20)
    emb = {n["entity_id"]: torch.tensor(n["text_embedding"]) for n in e["node_features"]}
    dde = {eid: torch.tensor(v) for eid, v in e["entity_dde"].items()}
    gd = assemble_micro_subgraph([t["triple"] for t in e["evidence_triples"]], emb, dde)
    rmap = {i: rid for i, rid in enumerate(sorted(e["relation_labels"]))} if e["relation_labels"] else None
    return (p, gd, emb, dde, e["relation_labels"], rmap,
            {n["entity_id"]: n["label"] for n in e["node_features"]})

# 加载训练好的模型
if MODEL_PATH.exists():
    model = torch.load(str(MODEL_PATH), map_location="cpu", weights_only=False)
    print("已加载模型:", MODEL_PATH.name)
else:
    model = RGCNNodeScorer(in_dim=138, hidden_dim=64, num_relations=4)
    print("使用随机权重")

# 预加载默认示例
_cache["aspirin"] = _load("aspirin")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/infer"):
            ex = parse_qs(urlparse(self.path).query).get("example", ["aspirin"])[0]
            if ex not in _cache:
                _cache[ex] = _load(ex)
            p, gd, emb, dde, rl, rmap, el = _cache[ex]
            result = run_micro_inference(model=model, graph_data=gd, topic_entity_ids=p["topic_entities"],
                entity_embeddings=emb, entity_dde=dde, entity_labels=el, relation_labels=rl,
                relation_id_map=rmap, top_k=5, max_hops=3)
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"question": p["question_text"],
                "candidates": result["candidate_answers"], "paths": result["reasoning_paths"]},
                ensure_ascii=False).encode("utf-8"))
            return

        if self.path == "/" or self.path.startswith("/?"):
            self.path = "/index.html"
        fp = FRONTEND_DIR / self.path.lstrip("/")
        if fp.is_file():
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(str(fp))[0] or "application/octet-stream")
            self.end_headers()
            with open(fp, "rb") as f:
                self.wfile.write(f.read())
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b"404")

    def log_message(self, format, *args):
        print(f"[server] {args[0]} {args[1]}")

if __name__ == "__main__":
    port = 8000
    print(f"Open http://localhost:{port} in your browser")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()


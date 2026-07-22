"""微观+GNN 演示服务器（纯内置模块，无需安装）"""

import sys, os, json, mimetypes
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

# 把 backend/ 加入路径
_BACKEND = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _BACKEND)

# 预加载模型
import torch
from retrieval.micro_rag import build_micro_evidence_subgraph, MicroRetriever
from gnn.core.model import RGCNNodeScorer
from gnn.micro_gnn.inference import run_micro_inference
from gnn.micro_gnn.loader import assemble_micro_subgraph

macro_path = Path(__file__).resolve().parents[2] / "backend" / "retrieval" / "examples" / "macro_subgraph_v0.1.json"
with open(macro_path, encoding="utf-8") as f:
    payload = json.load(f)

retriever = MicroRetriever()
evidence = build_micro_evidence_subgraph(payload, retriever, top_k=20)
triples = [et["triple"] for et in evidence["evidence_triples"]]
entity_embeddings = {nf["entity_id"]: torch.tensor(nf["text_embedding"]) for nf in evidence["node_features"]}
entity_labels = {nf["entity_id"]: nf["label"] for nf in evidence["node_features"]}
entity_dde = {eid: torch.tensor(vec) for eid, vec in evidence["entity_dde"].items()}
rel_labels = evidence["relation_labels"]
gd = assemble_micro_subgraph(triples, entity_embeddings, entity_dde)
relation_id_map = {i: rid for i, rid in enumerate(sorted(rel_labels))} if rel_labels else None
in_dim = evidence["feature_spec"]["gnn_input_dim"]
model = RGCNNodeScorer(in_dim=in_dim, hidden_dim=64, num_relations=gd.num_relations)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/api/infer":
            result = run_micro_inference(
                model=model, graph_data=gd, topic_entity_ids=payload["topic_entities"],
                entity_embeddings=entity_embeddings, entity_dde=entity_dde,
                entity_labels=entity_labels, relation_labels=rel_labels,
                relation_id_map=relation_id_map, top_k=5, max_hops=3,
            )
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({
                "question": payload["question_text"],
                "candidates": result["candidate_answers"],
                "paths": result["reasoning_paths"],
            }, ensure_ascii=False).encode("utf-8"))
            return

        # 静态文件
        if self.path == "/":
            self.path = "/index.html"
        file_path = FRONTEND_DIR / self.path.lstrip("/")
        if file_path.is_file():
            mime, _ = mimetypes.guess_type(str(file_path))
            self.send_response(200)
            self.send_header("Content-Type", mime or "application/octet-stream")
            self.end_headers()
            with open(file_path, "rb") as f:
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
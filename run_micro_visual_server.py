"""微观检索过程可视化服务器。

该入口独立于 backend/gnn/server.py，避免修改或占用队友维护的 GNN
演示服务器。它复用现有微观检索与 GNN 推理模块，并额外向前端返回
DDE、Top-K 证据评分和微观证据子图等可视化数据。
"""

import json
import mimetypes
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


ROOT_DIR = Path(__file__).resolve().parent
BACKEND_DIR = ROOT_DIR / "backend"
sys.path.insert(0, str(BACKEND_DIR))

import torch

from gnn.core.model import RGCNNodeScorer
from gnn.micro_gnn.inference import run_micro_inference
from gnn.micro_gnn.loader import assemble_micro_subgraph
from retrieval.micro_rag import MicroRetriever, build_micro_evidence_subgraph


EXAMPLES_DIR = BACKEND_DIR / "retrieval" / "examples"
MODEL_PATH = ROOT_DIR / "micro_model.pth"
FRONTEND_DIR = ROOT_DIR / "frontend"

retriever = MicroRetriever()
cache = {}


def load_example(name):
    """加载当前演示数据；以后可替换为实际问答流程的宏观子图输入。"""

    filename = {
        "aspirin": "macro_subgraph_v0.1.json",
        "einstein": "einstein_macro_subgraph_v0.1.json",
        "book": "book_macro_subgraph_v0.1.json",
    }.get(name, "macro_subgraph_v0.1.json")

    with open(EXAMPLES_DIR / filename, encoding="utf-8") as file:
        payload = json.load(file)

    evidence = build_micro_evidence_subgraph(payload, retriever, top_k=20)
    embeddings = {
        node["entity_id"]: torch.tensor(node["text_embedding"])
        for node in evidence["node_features"]
    }
    entity_dde = {
        entity_id: torch.tensor(vector)
        for entity_id, vector in evidence["entity_dde"].items()
    }
    graph_data = assemble_micro_subgraph(
        [item["triple"] for item in evidence["evidence_triples"]],
        embeddings,
        entity_dde,
    )
    relation_labels = evidence["relation_labels"]
    relation_id_map = (
        {
            index: relation_id
            for index, relation_id in enumerate(sorted(relation_labels))
        }
        if relation_labels
        else None
    )
    entity_labels = {
        node["entity_id"]: node["label"]
        for node in evidence["node_features"]
    }
    return (
        payload,
        evidence,
        graph_data,
        embeddings,
        entity_dde,
        relation_labels,
        relation_id_map,
        entity_labels,
    )


def load_model():
    model = RGCNNodeScorer(in_dim=138, hidden_dim=64, num_relations=4)
    if MODEL_PATH.exists():
        state_dict = torch.load(
            str(MODEL_PATH), map_location="cpu", weights_only=True
        )
        model.load_state_dict(state_dict, strict=False)
        print(f"已加载模型：{MODEL_PATH.name}")
    else:
        print("未找到 micro_model.pth，当前使用随机初始化模型")
    return model


model = load_model()


def build_response(example_name):
    if example_name not in cache:
        cache[example_name] = load_example(example_name)

    (
        payload,
        evidence,
        graph_data,
        embeddings,
        entity_dde,
        relation_labels,
        relation_id_map,
        entity_labels,
    ) = cache[example_name]

    inference = run_micro_inference(
        model=model,
        graph_data=graph_data,
        topic_entity_ids=payload["topic_entities"],
        entity_embeddings=embeddings,
        entity_dde=entity_dde,
        entity_labels=entity_labels,
        relation_labels=relation_labels,
        relation_id_map=relation_id_map,
        top_k=5,
        max_hops=3,
    )

    graph_nodes = [
        {
            "entity_id": node["entity_id"],
            "label": node["label"],
            "is_topic": node["entity_id"] in payload["topic_entities"],
            "dde": evidence["entity_dde"].get(node["entity_id"], []),
        }
        for node in evidence["node_features"]
    ]

    graph_edges = []
    for item in evidence["evidence_triples"]:
        head_id, relation_id, tail_id = item["triple"]
        graph_edges.append(
            {
                "head_id": head_id,
                "relation_id": relation_id,
                "relation_label": relation_labels.get(
                    relation_id, relation_id
                ),
                "tail_id": tail_id,
                "head_label": entity_labels.get(head_id, head_id),
                "tail_label": entity_labels.get(tail_id, tail_id),
                "relevance_score": item["relevance_score"],
                "head_dde": item["dde"]["head_dde"],
                "tail_dde": item["dde"]["tail_dde"],
            }
        )

    return {
        "question": payload["question_text"],
        "candidates": inference["candidate_answers"],
        "paths": inference["reasoning_paths"],
        "micro_retrieval": {
            "topic_entity_ids": payload["topic_entities"],
            "input_triple_count": len(
                payload["macro_subgraph"]["triples"]
            ),
            "selected_triple_count": len(graph_edges),
            "nodes": graph_nodes,
            "evidence_triples": graph_edges,
            "feature_spec": evidence["feature_spec"],
            "scoring": evidence["scoring"],
        },
    }


class Handler(BaseHTTPRequestHandler):
    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/micro-visualize":
            example_name = parse_qs(parsed.query).get(
                "example", ["aspirin"]
            )[0]
            try:
                self.send_json(build_response(example_name))
            except (ValueError, KeyError) as error:
                self.send_json({"error": str(error)}, status=400)
            return

        request_path = parsed.path
        if request_path == "/":
            request_path = "/index.html"
        file_path = FRONTEND_DIR / request_path.lstrip("/")
        if file_path.is_file():
            self.send_response(200)
            self.send_header(
                "Content-Type",
                mimetypes.guess_type(str(file_path))[0]
                or "application/octet-stream",
            )
            self.end_headers()
            with open(file_path, "rb") as file:
                self.wfile.write(file.read())
            return

        self.send_response(404)
        self.end_headers()
        self.wfile.write(b"404")

    def log_message(self, format_string, *args):
        print(f"[micro-visual] {args[0]} {args[1]}")


if __name__ == "__main__":
    port = 8001
    print(f"请在浏览器打开 http://localhost:{port}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()

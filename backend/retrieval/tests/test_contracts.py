import json
import unittest
from pathlib import Path

from micro_rag.contracts import build_micro_evidence_subgraph, parse_macro_subgraph
from micro_rag.retriever import MicroRetriever


ROOT = Path(__file__).resolve().parents[1]


class ContractV01Test(unittest.TestCase):
    def setUp(self):
        self.payload = json.loads(
            (ROOT / "examples" / "macro_subgraph_v0.1.json").read_text(encoding="utf-8")
        )

    def test_parses_id_graph_and_keeps_labels_for_semantics(self):
        question_id, question, topics, triples, nodes = parse_macro_subgraph(self.payload)
        self.assertEqual(question_id, "q_0001")
        self.assertIn("同时服用", question)
        self.assertEqual(topics, ["Q1024", "Q1088"])
        self.assertEqual(triples[0].head, "Q1024")
        self.assertEqual(triples[0].head_text, "阿司匹林")
        self.assertEqual(triples[0].relation_text, "增加风险")
        self.assertIn("Q2031", nodes)

    def test_output_matches_gnn_contract(self):
        result = build_micro_evidence_subgraph(
            self.payload, MicroRetriever(rounds=2), top_k=2
        )
        self.assertEqual(result["schema_version"], "0.1")
        self.assertEqual(result["question_id"], "q_0001")
        self.assertEqual(result["top_k"], 2)
        self.assertEqual(len(result["evidence_triples"]), 2)
        scores = [item["relevance_score"] for item in result["evidence_triples"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for item in result["evidence_triples"]:
            self.assertEqual(len(item["triple"]), 3)
            self.assertEqual(len(item["dde"]["head_dde"]), 10)
            self.assertEqual(len(item["dde"]["tail_dde"]), 10)
        selected_ids = {item["entity_id"] for item in result["node_features"]}
        triple_entity_ids = {
            entity_id
            for item in result["evidence_triples"]
            for entity_id in (item["triple"][0], item["triple"][2])
        }
        self.assertEqual(selected_ids, triple_entity_ids)
        self.assertEqual(result["feature_spec"]["dde_dim"], 10)
        self.assertEqual(result["feature_spec"]["text_embedding_dim"], 128)
        self.assertEqual(result["feature_spec"]["gnn_input_dim"], 138)
        self.assertEqual(result["scoring"]["scorer_type"], "heuristic")
        self.assertFalse(result["scoring"]["model_loaded"])
        self.assertEqual(set(result["entity_dde"]), selected_ids)
        self.assertEqual(len(result["relation_labels"]), 1)
        for feature in result["node_features"]:
            self.assertTrue(feature["label"])
            self.assertEqual(len(feature["text_embedding"]), 128)

    def test_real_embedding_id_is_preserved_without_fake_ids(self):
        self.payload["macro_subgraph"]["nodes"][0]["text_embedding_id"] = "shared/emb_Q1024"
        result = build_micro_evidence_subgraph(self.payload, top_k=4)
        features = {item["entity_id"]: item for item in result["node_features"]}
        self.assertEqual(features["Q1024"]["text_embedding_id"], "shared/emb_Q1024")
        self.assertNotIn("text_embedding_id", features["Q1088"])

    def test_formal_mode_requires_mlp_model(self):
        with self.assertRaisesRegex(ValueError, "没有加载模型"):
            build_micro_evidence_subgraph(self.payload, require_mlp=True)

    def test_rejects_unknown_schema_version(self):
        self.payload["schema_version"] = "9.9"
        with self.assertRaisesRegex(ValueError, "暂不支持"):
            build_micro_evidence_subgraph(self.payload)

    def test_rejects_topic_missing_from_nodes(self):
        self.payload["topic_entities"] = ["Q_NOT_FOUND"]
        with self.assertRaisesRegex(ValueError, "未出现在 nodes"):
            build_micro_evidence_subgraph(self.payload)


if __name__ == "__main__":
    unittest.main()

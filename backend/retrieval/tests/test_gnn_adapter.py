import json
import unittest
from pathlib import Path

from micro_rag.contracts import build_micro_evidence_subgraph
from micro_rag.gnn_adapter import prepare_gnn_handoff


ROOT = Path(__file__).resolve().parents[1]


class GNNAdapterTest(unittest.TestCase):
    def setUp(self):
        macro_payload = json.loads(
            (ROOT / "examples" / "macro_subgraph_v0.1.json").read_text(encoding="utf-8")
        )
        self.payload = build_micro_evidence_subgraph(macro_payload, top_k=3)

    def test_builds_valid_gnn_handoff(self):
        handoff = prepare_gnn_handoff(self.payload)
        self.assertEqual(handoff.gnn_input_dim, 138)
        self.assertEqual(len(handoff.triples), 3)
        self.assertEqual(handoff.relation_to_id, {"R017": 0})
        involved = {entity for triple in handoff.triples for entity in (triple[0], triple[2])}
        self.assertEqual(set(handoff.node_embeddings), involved)
        self.assertEqual(set(handoff.entity_dde), involved)
        self.assertTrue(all(len(value) == 128 for value in handoff.node_embeddings.values()))
        self.assertTrue(all(len(value) == 10 for value in handoff.entity_dde.values()))

    def test_rejects_wrong_embedding_dimension(self):
        self.payload["node_features"][0]["text_embedding"] = [0.0]
        with self.assertRaisesRegex(ValueError, "文本嵌入维度"):
            prepare_gnn_handoff(self.payload)

    def test_rejects_wrong_dde_dimension(self):
        entity_id = next(iter(self.payload["entity_dde"]))
        self.payload["entity_dde"][entity_id] = [0.0]
        with self.assertRaisesRegex(ValueError, "DDE 维度"):
            prepare_gnn_handoff(self.payload)

    def test_rejects_unsorted_evidence(self):
        self.payload["evidence_triples"][0]["relevance_score"] = 0.0
        self.payload["evidence_triples"][1]["relevance_score"] = 1.0
        with self.assertRaisesRegex(ValueError, "降序"):
            prepare_gnn_handoff(self.payload)


if __name__ == "__main__":
    unittest.main()

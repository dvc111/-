import json
import unittest
from pathlib import Path

from micro_rag import build_micro_evidence_subgraph, prepare_gnn_handoff


ROOT = Path(__file__).resolve().parents[1]


class MultiDomainExampleTest(unittest.TestCase):
    def _run_example(self, filename, expected_triples):
        payload = json.loads((ROOT / "examples" / filename).read_text(encoding="utf-8"))
        result = build_micro_evidence_subgraph(payload, top_k=2)
        selected = {tuple(item["triple"]) for item in result["evidence_triples"]}
        self.assertEqual(selected, set(expected_triples))
        handoff = prepare_gnn_handoff(result)
        self.assertEqual(handoff.gnn_input_dim, 138)
        self.assertEqual(len(handoff.triples), 2)

    def test_einstein_two_hop_question(self):
        self._run_example(
            "einstein_macro_subgraph_v0.1.json",
            {
                ("Q_EINSTEIN", "R_BIRTH_COUNTRY", "Q_GERMANY"),
                ("Q_GERMANY", "R_OFFICIAL_LANGUAGE", "Q_GERMAN"),
            },
        )

    def test_book_author_birthplace_question(self):
        self._run_example(
            "book_macro_subgraph_v0.1.json",
            {
                ("Q_THREE_BODY", "R_AUTHOR", "Q_LIU_CIXIN"),
                ("Q_LIU_CIXIN", "R_BIRTH_PLACE", "Q_BEIJING"),
            },
        )


if __name__ == "__main__":
    unittest.main()

import json
import tempfile
import unittest
from pathlib import Path

from micro_rag.dde import directional_distance_encoding
from micro_rag.models import Triple
from micro_rag.retriever import MicroRetriever
from micro_rag.training import train_from_jsonl
from micro_rag.weak_supervision import shortest_path_triple_ids


TRIPLES = [
    Triple("阿尔伯特·爱因斯坦", "出生国家", "德国", "t1"),
    Triple("德国", "官方语言", "德语", "t2"),
    Triple("阿尔伯特·爱因斯坦", "职业", "物理学家", "t3"),
    Triple("德国", "所属洲", "欧洲", "t4"),
]


class MicroRagTest(unittest.TestCase):
    def test_dde_respects_edge_direction(self):
        dde = directional_distance_encoding(TRIPLES, ["阿尔伯特·爱因斯坦"], rounds=2)
        self.assertEqual(dde["德国"][3], 1.0)  # 第 1 轮入边传播收到主题标记
        self.assertEqual(dde["德语"][5], 1.0)  # 第 2 轮入边传播收到主题标记
        self.assertEqual(dde["德国"][7], 0.0)  # 反向传播不应混淆方向

    def test_retrieve_returns_relevant_chain(self):
        result = MicroRetriever().retrieve(
            "爱因斯坦出生国家的官方语言是什么？",
            ["阿尔伯特·爱因斯坦"],
            TRIPLES,
            top_k=2,
        )
        selected = {item.triple.id for item in result.evidence}
        self.assertEqual(selected, {"t1", "t2"})

    def test_shortest_path_creates_weak_labels(self):
        labels = shortest_path_triple_ids(TRIPLES, ["阿尔伯特·爱因斯坦"], ["德语"])
        self.assertEqual(labels, {0, 1})

    def test_mlp_can_be_trained_and_loaded(self):
        record = {
            "question": "爱因斯坦出生国家的官方语言是什么？",
            "topic_entities": ["阿尔伯特·爱因斯坦"],
            "answer_entities": ["德语"],
            "triples": [
                {"head": item.head, "relation": item.relation, "tail": item.tail}
                for item in TRIPLES
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "train.jsonl"
            model_path = Path(directory) / "model.json"
            data_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")
            metrics = train_from_jsonl(data_path, model_path, epochs=30)
            self.assertTrue(model_path.exists())
            self.assertLess(metrics["final_loss"], metrics["initial_loss"])
            result = MicroRetriever(model_path=model_path).retrieve(
                record["question"], record["topic_entities"], TRIPLES, top_k=2
            )
            self.assertEqual(len(result.evidence), 2)


if __name__ == "__main__":
    unittest.main()


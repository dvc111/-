from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .mlp import TinyMLP
from .models import Triple
from .retriever import MicroRetriever
from .weak_supervision import shortest_path_triple_ids


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def train_from_jsonl(
    data_path: str | Path,
    output_path: str | Path,
    rounds: int = 2,
    epochs: int = 300,
) -> dict[str, float | int]:
    retriever = MicroRetriever(rounds=rounds)
    samples: list[tuple[list[float], int]] = []
    positive_count = 0
    for record in load_jsonl(data_path):
        triples = [Triple.from_dict(value, index) for index, value in enumerate(record["triples"])]
        positive_ids = shortest_path_triple_ids(
            triples, record["topic_entities"], record["answer_entities"]
        )
        features = retriever.feature_rows(record["question"], record["topic_entities"], triples)
        for index, row in enumerate(features):
            label = int(index in positive_ids)
            positive_count += label
            samples.append((row, label))
    if not samples:
        raise ValueError("训练文件中没有三元组")
    if positive_count == 0:
        raise ValueError("没有找到主题实体到答案实体的最短路径")
    model = TinyMLP(input_size=len(samples[0][0]))
    losses = model.fit(samples, epochs=epochs)
    model.save(output_path)
    return {
        "samples": len(samples),
        "positive_samples": positive_count,
        "initial_loss": round(losses[0], 6),
        "final_loss": round(losses[-1], 6),
    }


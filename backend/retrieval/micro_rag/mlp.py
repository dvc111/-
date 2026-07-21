from __future__ import annotations

import json
import math
import random
from pathlib import Path


def _sigmoid(value: float) -> float:
    value = max(-30.0, min(30.0, value))
    return 1.0 / (1.0 + math.exp(-value))


class TinyMLP:
    """纯 Python 两层 MLP，用二分类交叉熵学习证据三元组。"""

    def __init__(self, input_size: int, hidden_size: int = 12, seed: int = 42):
        randomizer = random.Random(seed)
        scale = 1.0 / math.sqrt(input_size)
        self.w1 = [
            [randomizer.uniform(-scale, scale) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.b1 = [0.0] * hidden_size
        self.w2 = [randomizer.uniform(-scale, scale) for _ in range(hidden_size)]
        self.b2 = 0.0

    @property
    def input_size(self) -> int:
        return len(self.w1[0])

    def _forward(self, features: list[float]) -> tuple[list[float], float]:
        if len(features) != self.input_size:
            raise ValueError("MLP 输入维度不一致")
        hidden = [
            max(0.0, sum(weight * value for weight, value in zip(row, features)) + bias)
            for row, bias in zip(self.w1, self.b1)
        ]
        probability = _sigmoid(sum(w * h for w, h in zip(self.w2, hidden)) + self.b2)
        return hidden, probability

    def predict(self, features: list[float]) -> float:
        return self._forward(features)[1]

    def fit(
        self,
        samples: list[tuple[list[float], int]],
        epochs: int = 300,
        learning_rate: float = 0.05,
        seed: int = 42,
    ) -> list[float]:
        if not samples:
            raise ValueError("训练样本不能为空")
        randomizer = random.Random(seed)
        losses: list[float] = []
        for _ in range(epochs):
            randomizer.shuffle(samples)
            total_loss = 0.0
            for features, label in samples:
                hidden, probability = self._forward(features)
                total_loss -= label * math.log(probability + 1e-9) + (1 - label) * math.log(
                    1 - probability + 1e-9
                )
                output_gradient = probability - label
                old_w2 = list(self.w2)
                for j, value in enumerate(hidden):
                    self.w2[j] -= learning_rate * output_gradient * value
                self.b2 -= learning_rate * output_gradient
                for j, value in enumerate(hidden):
                    if value <= 0:
                        continue
                    hidden_gradient = output_gradient * old_w2[j]
                    for i, feature in enumerate(features):
                        self.w1[j][i] -= learning_rate * hidden_gradient * feature
                    self.b1[j] -= learning_rate * hidden_gradient
            losses.append(total_loss / len(samples))
        return losses

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(
                {"w1": self.w1, "b1": self.b1, "w2": self.w2, "b2": self.b2},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TinyMLP":
        values = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(len(values["w1"][0]), len(values["w1"]))
        model.w1, model.b1 = values["w1"], values["b1"]
        model.w2, model.b2 = values["w2"], values["b2"]
        return model


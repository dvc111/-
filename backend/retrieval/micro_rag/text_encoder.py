from __future__ import annotations

import hashlib
import math
import re


_LATIN_OR_NUMBER = re.compile(r"[a-z0-9_]+")
_CJK_BLOCK = re.compile(r"[\u3400-\u9fff]+")


def tokenize(text: str) -> list[str]:
    """同时支持中英文的无模型分词，适合离线演示。"""

    normalized = text.casefold().replace("_", " ")
    tokens = _LATIN_OR_NUMBER.findall(normalized)
    for block in _CJK_BLOCK.findall(normalized):
        tokens.extend(block)
        tokens.extend(block[i : i + 2] for i in range(len(block) - 1))
    return tokens


class HashingTextEncoder:
    """确定性哈希文本编码器，可替换为 BGE/GTE 等预训练编码器。"""

    def __init__(self, dimensions: int = 128):
        if dimensions < 16:
            raise ValueError("dimensions 至少为 16")
        self.dimensions = dimensions

    def encode(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            value = int.from_bytes(digest, "little")
            index = value % self.dimensions
            sign = 1.0 if (value >> 8) & 1 else -1.0
            vector[index] += sign
        norm = math.sqrt(sum(value * value for value in vector))
        return vector if norm == 0 else [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("向量维度不一致")
    # 编码器已归一化；映射到 [0, 1] 方便解释与融合。
    raw = sum(a * b for a, b in zip(left, right))
    return max(0.0, min(1.0, (raw + 1.0) / 2.0))


def token_overlap(left: str, right: str) -> float:
    left_set, right_set = set(tokenize(left)), set(tokenize(right))
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / math.sqrt(len(left_set) * len(right_set))


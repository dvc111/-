from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable

from .models import Triple


def _mean(vectors: list[list[float]], width: int) -> list[float]:
    if not vectors:
        return [0.0] * width
    return [sum(vector[i] for vector in vectors) / len(vectors) for i in range(width)]


def directional_distance_encoding(
    triples: Iterable[Triple], topic_entities: Iterable[str], rounds: int = 2
) -> dict[str, list[float]]:
    """计算论文中的 DDE：沿入边、出边分别进行多轮均值传播。

    核心创新点：两个方向从同一主题实体标记独立传播，保留“从主题实体
    到当前节点”和“从当前节点返回主题实体”的非对称结构信息；各轮结果
    不覆盖而是依次拼接，因此轻量 MLP 仍能感知多跳层次。
    """

    if rounds < 0:
        raise ValueError("rounds 不能为负数")
    triples = list(triples)
    entities = {value for triple in triples for value in (triple.head, triple.tail)}
    entities.update(topic_entities)
    topics = set(topic_entities)

    incoming: dict[str, list[str]] = defaultdict(list)
    outgoing: dict[str, list[str]] = defaultdict(list)
    for triple in triples:
        incoming[triple.tail].append(triple.head)
        outgoing[triple.head].append(triple.tail)

    seed = {entity: [0.0, 1.0] if entity in topics else [1.0, 0.0] for entity in entities}
    parts = {entity: list(seed[entity]) for entity in entities}

    forward = seed
    for _ in range(rounds):
        forward = {
            entity: _mean([forward[source] for source in incoming[entity]], 2)
            for entity in entities
        }
        for entity in entities:
            parts[entity].extend(forward[entity])

    reverse = seed
    for _ in range(rounds):
        reverse = {
            entity: _mean([reverse[target] for target in outgoing[entity]], 2)
            for entity in entities
        }
        for entity in entities:
            parts[entity].extend(reverse[entity])

    return parts


def topic_reachability(encoding: list[float]) -> float:
    """取各传播轮次中的 topic 通道最大值，作为可解释的结构相关度。"""

    return max(encoding[1::2], default=0.0)

from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable

from .models import Triple


def shortest_path_triple_ids(
    triples: list[Triple], topic_entities: Iterable[str], answer_entities: Iterable[str]
) -> set[int]:
    """用主题实体到答案实体的最短路径生成弱监督正样本。

    核心创新点：不要求人工逐条标注证据，利用已知问答对在局部子图上的
    最短路径自动构造正样本，其余候选作为负样本训练轻量 MLP。
    """

    adjacency: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for index, triple in enumerate(triples):
        # 与原项目一致，同时考虑正向和反向可达路径。
        adjacency[triple.head].append((triple.tail, index))
        adjacency[triple.tail].append((triple.head, index))

    answers = set(answer_entities)
    positive_ids: set[int] = set()
    for topic in topic_entities:
        queue = deque([topic])
        distance = {topic: 0}
        parents: dict[str, list[tuple[str, int]]] = defaultdict(list)
        nearest_answer_distance: int | None = None
        reached_answers: list[str] = []
        while queue:
            node = queue.popleft()
            if nearest_answer_distance is not None and distance[node] > nearest_answer_distance:
                break
            if node in answers and node != topic:
                nearest_answer_distance = distance[node]
                reached_answers.append(node)
                continue
            for neighbor, triple_id in adjacency[node]:
                candidate_distance = distance[node] + 1
                if neighbor not in distance:
                    distance[neighbor] = candidate_distance
                    parents[neighbor].append((node, triple_id))
                    queue.append(neighbor)
                elif distance[neighbor] == candidate_distance:
                    parents[neighbor].append((node, triple_id))

        stack = list(reached_answers)
        seen = set(stack)
        while stack:
            node = stack.pop()
            for parent, triple_id in parents[node]:
                positive_ids.add(triple_id)
                if parent not in seen and parent != topic:
                    seen.add(parent)
                    stack.append(parent)
    return positive_ids

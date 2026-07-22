from __future__ import annotations

import math
from collections.abc import Iterable
from pathlib import Path

from .dde import directional_distance_encoding, topic_reachability
from .mlp import TinyMLP
from .models import Evidence, RetrievalResult, Triple
from .text_encoder import HashingTextEncoder, cosine, token_overlap


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, value))))


class MicroRetriever:
    """DDE + 语义特征 + 轻量评分器的微观证据检索器。"""

    def __init__(
        self,
        rounds: int = 2,
        embedding_dimensions: int = 128,
        model_path: str | Path | None = None,
    ):
        self.rounds = rounds
        self.encoder = HashingTextEncoder(embedding_dimensions)
        self.model = TinyMLP.load(model_path) if model_path else None

    @property
    def scorer_type(self) -> str:
        """返回当前实际使用的评分器，避免把规则评分误称为 MLP。"""

        return "mlp" if self.model is not None else "heuristic"

    def _features(
        self,
        question: str,
        question_vector: list[float],
        triple: Triple,
        head_dde: list[float],
        tail_dde: list[float],
    ) -> tuple[list[float], dict[str, float]]:
        semantic = cosine(question_vector, self.encoder.encode(triple.text()))
        relation = max(
            cosine(question_vector, self.encoder.encode(triple.relation_text)),
            token_overlap(question, triple.relation_text),
        )
        endpoint = max(
            token_overlap(question, triple.head_text),
            token_overlap(question, triple.tail_text),
        )
        structure = max(topic_reachability(head_dde), topic_reachability(tail_dde))
        # 核心创新点：把问题-三元组语义相关度与主题实体中心的 DDE
        # 结构可达性拼接，让轻量评分器也能识别多跳证据方向。
        compact = [semantic, relation, endpoint, structure]
        features = compact + head_dde + tail_dde
        components = {
            "semantic_score": semantic,
            "relation_score": relation,
            "endpoint_score": endpoint,
            "structure_score": structure,
        }
        return features, components

    @staticmethod
    def _fallback_score(components: dict[str, float]) -> float:
        # 未训练时也能用于原型展示；每一项都可在返回结果中审计。
        logit = (
            -2.1
            # 问题中的关系约束比主题实体字面重合更能区分多跳分支；
            # 例如“作者出生地”不能被主题实体直连的“作品类型”抢占。
            + 1.5 * components["semantic_score"]
            + 4.0 * components["relation_score"]
            + 0.4 * components["endpoint_score"]
            + 0.8 * components["structure_score"]
        )
        return _sigmoid(logit)

    def feature_rows(
        self, question: str, topic_entities: Iterable[str], triples: list[Triple]
    ) -> list[list[float]]:
        topic_entities = tuple(topic_entities)
        dde = directional_distance_encoding(triples, topic_entities, self.rounds)
        question_vector = self.encoder.encode(question)
        return [
            self._features(
                question,
                question_vector,
                triple,
                dde[triple.head],
                dde[triple.tail],
            )[0]
            for triple in triples
        ]

    def retrieve(
        self,
        question: str,
        topic_entities: Iterable[str],
        triples: list[Triple],
        top_k: int = 5,
        threshold: float = 0.0,
        precomputed_dde: dict[str, list[float]] | None = None,
    ) -> RetrievalResult:
        if not question.strip():
            raise ValueError("question 不能为空")
        if top_k <= 0:
            raise ValueError("top_k 必须大于 0")
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("threshold 必须在 0 到 1 之间")
        topic_entities = tuple(dict.fromkeys(topic_entities))
        if not topic_entities:
            raise ValueError("至少需要一个主题实体")
        dde = precomputed_dde or directional_distance_encoding(
            triples, topic_entities, self.rounds
        )
        question_vector = self.encoder.encode(question)
        evidence: list[Evidence] = []
        # 核心创新点：三元组相互独立打分，可一次性并行筛选任意形态子图，
        # 不受逐跳贪心路径搜索约束；这里保留简单循环以兼容纯 Python 环境。
        for triple in triples:
            features, components = self._features(
                question,
                question_vector,
                triple,
                dde[triple.head],
                dde[triple.tail],
            )
            score = self.model.predict(features) if self.model else self._fallback_score(components)
            if score >= threshold:
                evidence.append(Evidence(triple=triple, score=score, **components))
        evidence.sort(key=lambda item: (-item.score, item.triple.id))
        return RetrievalResult(
            question=question,
            topic_entities=topic_entities,
            evidence=tuple(evidence[: min(top_k, len(evidence))]),
            all_candidate_count=len(triples),
        )

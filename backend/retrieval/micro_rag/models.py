from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Triple:
    """知识图谱中的一条有向三元组。"""

    head: str
    relation: str
    tail: str
    id: str = ""
    head_label: str | None = None
    relation_label: str | None = None
    tail_label: str | None = None

    @classmethod
    def from_dict(cls, value: dict[str, Any], index: int = 0) -> "Triple":
        return cls(
            head=str(value["head"]),
            relation=str(value["relation"]),
            tail=str(value["tail"]),
            id=str(value.get("id") or f"t{index}"),
            head_label=value.get("head_label"),
            relation_label=value.get("relation_label"),
            tail_label=value.get("tail_label"),
        )

    def text(self) -> str:
        return f"{self.head_text} {self.relation_text} {self.tail_text}"

    @property
    def head_text(self) -> str:
        return self.head_label or self.head

    @property
    def relation_text(self) -> str:
        return self.relation_label or self.relation

    @property
    def tail_text(self) -> str:
        return self.tail_label or self.tail


@dataclass(frozen=True, slots=True)
class Evidence:
    triple: Triple
    score: float
    semantic_score: float
    relation_score: float
    structure_score: float
    endpoint_score: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["score"] = round(self.score, 6)
        for key in (
            "semantic_score",
            "relation_score",
            "structure_score",
            "endpoint_score",
        ):
            result[key] = round(result[key], 6)
        return result


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    question: str
    topic_entities: tuple[str, ...]
    evidence: tuple[Evidence, ...]
    all_candidate_count: int

    def to_dict(self) -> dict[str, Any]:
        entities = sorted(
            {name for item in self.evidence for name in (item.triple.head, item.triple.tail)}
        )
        return {
            "question": self.question,
            "topic_entities": list(self.topic_entities),
            "all_candidate_count": self.all_candidate_count,
            "selected_count": len(self.evidence),
            "evidence_subgraph": {
                "entities": entities,
                "triples": [item.to_dict() for item in self.evidence],
            },
        }

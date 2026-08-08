from __future__ import annotations

import os
from typing import Any

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field
except ImportError as error:  # pragma: no cover - 只在未安装 API 依赖时触发
    raise RuntimeError("请先运行 pip install -e .[api] 安装 HTTP 接口依赖") from error

from .models import Triple
from .retriever import MicroRetriever
from .contracts import build_micro_evidence_subgraph


class TripleBody(BaseModel):
    id: str | None = None
    head: str
    relation: str
    tail: str


class RetrieveBody(BaseModel):
    question: str = Field(min_length=1)
    topic_entities: list[str] = Field(min_length=1)
    triples: list[TripleBody]
    top_k: int = Field(default=5, ge=1, le=1000)
    threshold: float = Field(default=0.0, ge=0.0, le=1.0)


app = FastAPI(title="SubgraphRAG 微观检索服务", version="0.1.0")
retriever = MicroRetriever(model_path=os.getenv("MICRO_RAG_MODEL") or None)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/v1/micro-retrieve")
def micro_retrieve(body: RetrieveBody) -> dict[str, Any]:
    try:
        triples = [
            Triple.from_dict(
                (
                    item.model_dump(exclude_none=True)
                    if hasattr(item, "model_dump")
                    else item.dict(exclude_none=True)
                ),
                index,
            )
            for index, item in enumerate(body.triples)
        ]
        return retriever.retrieve(
            body.question,
            body.topic_entities,
            triples,
            body.top_k,
            body.threshold,
        ).to_dict()
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/api/v1/micro-evidence-subgraph")
def micro_evidence_subgraph(
    body: dict[str, Any],
    top_k: int = 20,
    threshold: float = 0.0,
    require_mlp: bool = False,
) -> dict[str, Any]:
    """团队联调接口：macro_subgraph -> micro_evidence_subgraph。"""

    try:
        if top_k < 1 or top_k > 1000:
            raise ValueError("top_k 必须在 1 到 1000 之间")
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("threshold 必须在 0 到 1 之间")
        return build_micro_evidence_subgraph(
            body, retriever, top_k, threshold, require_mlp
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

# -*- coding: utf-8 -*-
"""
超关系语义评分模块

核心创新点：
1. 将传统 KGQA 中逐 relation 搜索改为 super-relation 粒度检索；
2. 使用语义评分保留多个潜在推理方向，扩大宏观搜索空间，降低错误路径导致的检索失败。

"""

import re
from typing import List, Dict

import config
from prompt_list import build_hyper_relation_prompt
from kg_store import KGStore

# "{标签 (Score: 0.x)}"
_SCORE_PATTERN = re.compile(r"\{\s*(?P<label>[^{}()]+?)\s*\(Score:\s*(?P<score>[0-9.]+)\)\s*\}")


def _parse_scores(llm_output: str, candidate_labels: List[str]) -> Dict[str, float]:
    """解析 LLM 输出中的 {标签 (Score: x)} 片段，只保留候选集合中真实存在的标签。"""
    scores = {}
    for match in _SCORE_PATTERN.finditer(llm_output):
        label = match.group("label").strip()
        try:
            score = float(match.group("score"))
        except ValueError:
            continue
        if label in candidate_labels:
            scores[label] = max(0.0, min(1.0, score))
    return scores


def score_hyper_relations(question: str, kg: KGStore, llm_fn,
                           top_k: int = config.TOP_K_HYPER_RELATIONS) -> List[Dict]:
    """
    对 KG 中全部超关系打分并取 top_k
    """
    candidates = kg.all_hyper_relations()
    if not candidates:
        return []

    prompt = build_hyper_relation_prompt(question, candidates)
    llm_output = llm_fn(prompt)

    labels = [c["label"] for c in candidates]
    parsed = _parse_scores(llm_output, labels)

    # 若解析失败（例如本地小模型输出格式不稳定），退化为均匀打分，保证流程不中断，
    if not parsed:
        parsed = {label: 1.0 / len(labels) for label in labels}

    scored = [
        {
            "relation_id": c["hyper_relation_id"],
            "label": c["label"],
            "score": round(float(parsed.get(c["label"], 0.0)), 4),
        }
        for c in candidates
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]

# -*- coding: utf-8 -*-
"""
超关系语义评分模块（对应模块接口文档 1. 中 selected_hyper_relations 字段）。

直接继承原 REKNOS 项目 freebase_func.py 中
extract_relation_prompt + clean_relations() 的“LLM 打分 -> 正则解析 -> 排序取 top-K”
方法思想，只是把打分对象从具体 relation 换成了 super-relation（超关系），
这正是论文 Reasoning of LLMs over KGs with Super-Relations 的核心概念，
也是模块接口文档里“超关系语义评分”一步要做的事情。
"""

import re
from typing import List, Dict

import config
from prompt_list import build_hyper_relation_prompt
from kg_store import KGStore

# 复用原版 clean_relations 的正则风格： "{标签 (Score: 0.x)}"
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
    对 KG 中全部超关系打分并取 top_k，返回：
    [{"relation_id":..., "label":..., "score":...}, ...]，与接口文档字段一致。
    """
    candidates = kg.all_hyper_relations()
    if not candidates:
        return []

    prompt = build_hyper_relation_prompt(question, candidates)
    llm_output = llm_fn(prompt)

    labels = [c["label"] for c in candidates]
    parsed = _parse_scores(llm_output, labels)

    # 若解析失败（例如本地小模型输出格式不稳定），退化为均匀打分，保证流程不中断，
    # 这与原版 utils.clean_scores() 在解析失败时 "All entities are created equal" 的兜底思路一致。
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

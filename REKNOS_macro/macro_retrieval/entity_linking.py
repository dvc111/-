# -*- coding: utf-8 -*-
"""
实体链接模块（对应模块接口文档 1. 中 entity_linking / topic_entities 字段）。

延续原 REKNOS 项目的整体思路：先由 LLM 承担语言理解部分（这里是抽取问题中的
实体提及），再用可控的规则/相似度计算做确定性的消歧和打分，
而不是把“判断具体实体 id”这种确定性任务也丢给 LLM 自由发挥
（原版对关系打分也是类似的“LLM 提议 + 规则/正则解析”组合方式）。

流程：
1. LLM 从问题中抽取候选 mention（prompt_list.ENTITY_MENTION_EXTRACT_PROMPT）
2. 对每个 mention，在本地 KG 的别名索引中做字符串相似度匹配，找最佳实体
3. 相似度即作为 confidence，低于阈值的 mention 判定为链接失败并丢弃
"""

import difflib
from typing import List, Dict

import config
from prompt_list import ENTITY_MENTION_EXTRACT_PROMPT
from kg_store import KGStore


def extract_mentions(question: str, llm_fn) -> List[str]:
    """调用 LLM 抽取问题中的实体提及，每行一个。"""
    prompt = ENTITY_MENTION_EXTRACT_PROMPT.format(question=question)
    raw = llm_fn(prompt)
    mentions = [line.strip("：: 　\t-•").strip() for line in raw.splitlines()]
    mentions = [m for m in mentions if m and len(m) <= 20]
    # 去重，保持顺序
    seen = set()
    ordered = []
    for m in mentions:
        if m not in seen:
            seen.add(m)
            ordered.append(m)
    return ordered


def _best_alias_match(mention: str, kg: KGStore):
    """在别名表中找到与 mention 最相似的实体，返回 (entity_id, confidence) 或 (None, 0.0)。"""
    best_entity_id, best_score = None, 0.0
    for alias, entity_ids in kg.alias_index.items():
        score = difflib.SequenceMatcher(None, mention, alias).ratio()
        # 完全包含关系（如 mention 是 alias 的子串或反之）给予提升，缓解简单缩写/别名问题
        if mention in alias or alias in mention:
            score = max(score, 0.9)
        if score > best_score:
            best_score = score
            best_entity_id = entity_ids[0]
    return best_entity_id, best_score


def link_entities(question: str, kg: KGStore, llm_fn,
                   min_sim: float = config.ENTITY_LINK_MIN_SIM) -> List[Dict]:
    """
    执行实体链接，返回结构与接口文档一致的列表：
    [{"mention":..., "entity_id":..., "confidence":...}, ...]
    """
    mentions = extract_mentions(question, llm_fn)

    linked = []
    for mention in mentions:
        entity_id, confidence = _best_alias_match(mention, kg)
        if entity_id is not None and confidence >= min_sim:
            linked.append({
                "mention": mention,
                "entity_id": entity_id,
                "confidence": round(float(confidence), 4),
            })
    return linked


def fallback_alias_scan(question: str, kg: KGStore) -> List[Dict]:
    """
    兜底方案：当 LLM 不可用（如未启动 Ollama）时，直接在问题原文中做
    别名子串扫描，保证宏观检索流程在没有 LLM 的情况下也能跑通用于测试。
    """
    linked = []
    seen_entities = set()
    for alias, entity_ids in kg.alias_index.items():
        if alias and alias in question:
            entity_id = entity_ids[0]
            if entity_id in seen_entities:
                continue
            seen_entities.add(entity_id)
            linked.append({
                "mention": alias,
                "entity_id": entity_id,
                "confidence": 0.95 if alias == kg.entity_label(entity_id) else 0.85,
            })
    return linked

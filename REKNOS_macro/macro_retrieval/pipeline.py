# -*- coding: utf-8 -*-
"""
宏观检索模块总装配（软著一：宏观检索 + GNN 中的“宏观检索”部分，不含 GNN）。

对应模块接口文档《1. Agent + 宏观检索模块 -> GNN推理模块》中的 macro_subgraph 接口，
产出字段：schema_version / question_id / question_text / entity_linking /
topic_entities / selected_hyper_relations / macro_subgraph / max_hops。

三个子步骤：
1. entity_linking.link_entities        —— 实体链接
2. hyper_relation_scoring.score_hyper_relations —— 超关系语义评分
3. macro_subgraph.prune_macro_subgraph —— 宏观子图裁剪
均延续原 REKNOS 项目"LLM 提议 + 确定性规则解析/打分"的组合方法思想。
"""

from typing import Optional, Callable

import config
from kg_store import KGStore
from llm_client import run_llm, LLMError
import entity_linking
import hyper_relation_scoring
import macro_subgraph


def macro_retrieval(question_id: str,
                     question_text: str,
                     kg: KGStore,
                     llm_fn: Optional[Callable[[str], str]] = None,
                     top_k_hyper_relations: int = config.TOP_K_HYPER_RELATIONS,
                     max_hops: int = config.MAX_HOPS,
                     use_llm_fallback: bool = True) -> dict:
    """
    执行完整的宏观检索流程，返回与接口文档一致的 dict。

    llm_fn: 可注入自定义 LLM 调用（用于离线单元测试）；默认使用本地 Ollama/Phi-3。
    use_llm_fallback: 当 LLM 不可用时，实体链接退化为本地别名子串扫描
                       （见 entity_linking.fallback_alias_scan），
                       以保证在未启动 Ollama 的环境下宏观检索流程仍可跑通做结构验证；
                       超关系评分在 LLM 不可用时无法进行语义打分，会直接抛出异常。
    """
    if llm_fn is None:
        llm_fn = run_llm

    # ---- 1. 实体链接 ----
    try:
        linked = entity_linking.link_entities(question_text, kg, llm_fn)
    except LLMError:
        if not use_llm_fallback:
            raise
        linked = entity_linking.fallback_alias_scan(question_text, kg)

    topic_entities = [e["entity_id"] for e in linked]

    # ---- 2. 超关系语义评分 ----
    selected_hyper_relations = hyper_relation_scoring.score_hyper_relations(
        question_text, kg, llm_fn, top_k=top_k_hyper_relations
    )

    # ---- 3. 宏观子图裁剪 ----
    macro_sub = macro_subgraph.prune_macro_subgraph(
        topic_entities=topic_entities,
        entity_linking=linked,
        selected_hyper_relations=selected_hyper_relations,
        kg=kg,
        max_hops=max_hops,
    )

    return {
        "schema_version": config.SCHEMA_VERSION,
        "question_id": question_id,
        "question_text": question_text,

        "entity_linking": linked,
        "topic_entities": topic_entities,

        "selected_hyper_relations": selected_hyper_relations,

        "macro_subgraph": macro_sub,

        "max_hops": max_hops,
    }

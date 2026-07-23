# -*- coding: utf-8 -*-
"""
离线测试：用一个确定性的 mock LLM 函数替代真实 Phi-3 调用，
验证宏观检索三个子模块（实体链接 / 超关系评分 / 宏观子图裁剪）
组装出的 JSON 是否符合模块接口文档中 macro_subgraph 接口的字段结构。

运行： python test_pipeline.py
"""

import json

from kg_store import KGStore
from pipeline import macro_retrieval
import config


def mock_llm(prompt: str) -> str:
    """根据 prompt 中出现的关键词，返回符合各自解析格式的确定性文本。"""
    if "提及" in prompt and "候选超关系" not in prompt:
        # 实体提及抽取：从 prompt 末尾的问题里，用本地 KG 词表反查一次
        # （测试目的：不依赖真实语言模型也能抽出“阿司匹林 / 布洛芬”）
        return "阿司匹林\n布洛芬"

    if "候选超关系" in prompt:
        return (
            "{药物相互作用 (Score: 0.92)}: 两药合用的核心风险来自相互作用。\n"
            "{禁忌症 (Score: 0.81)}: 也可能触发共同的禁忌症。\n"
            "{适应症 (Score: 0.15)}: 与本问题关系较弱。\n"
            "{药物分类 (Score: 0.10)}: 与本问题关系较弱。\n"
        )

    return ""


def test_macro_retrieval_schema():
    kg = KGStore(config.DEFAULT_KG_PATH)
    result = macro_retrieval(
        question_id="q_0001",
        question_text="阿司匹林和布洛芬同时服用会有什么风险？",
        kg=kg,
        llm_fn=mock_llm,
        top_k_hyper_relations=2,
        max_hops=2,
    )

    # ---- 顶层字段 ----
    for key in ["schema_version", "question_id", "question_text",
                "entity_linking", "topic_entities",
                "selected_hyper_relations", "macro_subgraph", "max_hops"]:
        assert key in result, f"缺少字段: {key}"

    # ---- 实体链接 ----
    assert len(result["entity_linking"]) == 2
    mentions = {e["mention"] for e in result["entity_linking"]}
    assert mentions == {"阿司匹林", "布洛芬"}
    for e in result["entity_linking"]:
        assert 0.0 <= e["confidence"] <= 1.0
    assert set(result["topic_entities"]) == {"Q1024", "Q1088"}

    # ---- 超关系评分 ----
    assert len(result["selected_hyper_relations"]) == 2
    top_labels = [hr["label"] for hr in result["selected_hyper_relations"]]
    assert top_labels[0] == "药物相互作用"
    assert result["selected_hyper_relations"][0]["score"] >= result["selected_hyper_relations"][1]["score"]

    # ---- 宏观子图 ----
    nodes = result["macro_subgraph"]["nodes"]
    triples = result["macro_subgraph"]["triples"]
    node_ids = {n["entity_id"] for n in nodes}
    assert {"Q1024", "Q1088"}.issubset(node_ids)          # 主题实体必须在子图中
    assert "Q2031" in node_ids                              # 胃肠道出血应通过 R017/药物相互作用 被检索到
    for h, r, t in triples:
        assert h in node_ids and t in node_ids
        assert kg.hyper_relation_of(r) in {"HR03", "HR07"}   # 只允许被选中的两个超关系

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n[OK] 输出结构与模块接口文档 macro_subgraph 接口一致，全部断言通过。")


if __name__ == "__main__":
    test_macro_retrieval_schema()

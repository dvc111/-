# -*- coding: utf-8 -*-
"""
本地知识图谱存取层。

原 REKNOS 项目通过 SPARQL 查询 Freebase / Wikidata（freebase_func.py）。
按本次任务约束（不使用 Freebase/Wikidata、不下载大型 KG），这里替换为
一个加载本地 JSON 小型 KG 的内存索引，对外提供与原版语义类似的查询能力：
- 根据实体 id 找邻居三元组（对应原版 relation_search_prune 系列函数）
- 根据超关系 id 找到其下属具体关系（对应超关系分组）
- 实体名 / 别名索引（用于实体链接）
"""

import json
from collections import defaultdict


class KGStore:
    def __init__(self, kg_path: str):
        with open(kg_path, encoding="utf-8") as f:
            raw = json.load(f)

        self.entities = raw["entities"]              # entity_id -> {label, type, aliases}
        self.relations = raw["relations"]             # relation_id -> {label, hyper_relation_id}
        self.hyper_relations = raw["hyper_relations"]  # hyper_relation_id -> {label, relations}
        self.triples = [tuple(t) for t in raw["triples"]]  # [(head, relation, tail), ...]

        # 别名 -> [entity_id, ...]，用于实体链接的候选召回
        self.alias_index = defaultdict(list)
        for entity_id, info in self.entities.items():
            for alias in info.get("aliases", [info["label"]]):
                self.alias_index[alias].append(entity_id)

        # entity_id -> 出边三元组 list[(relation_id, tail_id)]
        self.out_edges = defaultdict(list)
        for h, r, t in self.triples:
            self.out_edges[h].append((r, t))

        # entity_id -> 度数（简单中心性度量，供 initial_importance 使用）
        degree = defaultdict(int)
        for h, r, t in self.triples:
            degree[h] += 1
            degree[t] += 1
        self.degree = degree
        self.max_degree = max(degree.values()) if degree else 1

    # ---- 基础查询 ----
    def entity_label(self, entity_id: str) -> str:
        return self.entities.get(entity_id, {}).get("label", entity_id)

    def all_aliases(self):
        return list(self.alias_index.keys())

    def hyper_relation_of(self, relation_id: str) -> str:
        return self.relations.get(relation_id, {}).get("hyper_relation_id")

    def hyper_relation_label(self, hyper_relation_id: str) -> str:
        return self.hyper_relations.get(hyper_relation_id, {}).get("label", hyper_relation_id)

    def all_hyper_relations(self):
        """返回 [{hyper_relation_id, label}, ...]，供超关系打分模块展示候选。"""
        return [
            {"hyper_relation_id": hr_id, "label": info["label"]}
            for hr_id, info in self.hyper_relations.items()
        ]

    def neighbors(self, entity_id: str):
        """返回 entity_id 的出边 [(relation_id, tail_id), ...]"""
        return self.out_edges.get(entity_id, [])

    def normalized_degree(self, entity_id: str) -> float:
        return self.degree.get(entity_id, 0) / self.max_degree if self.max_degree else 0.0

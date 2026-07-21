"""路径转自然语言。

将路径 [实体, 关系, 实体, 关系, ...] 转为
"实体1 → 关系1 → 实体2 → 关系2 → 答案"
格式的文本，供 LLM 读取。"""

from __future__ import annotations

from typing import Any


def verbalize(path: list[dict[str, Any]]) -> str:
    """将结构化路径转为自然语言字符串。

    Args:
        path: bfs.extract_shortest_path 输出的 path 字段，
              每个元素为 {"entity_id": str, ...} 或 {"relation_id": str, ...}。

    Returns:
        形如 "阿司匹林 → 增加风险 → 胃肠道出血" 的字符串。
        有 label 则优先用 label，否则 fallback 到 ID。
    """
    segments: list[str] = []
    for node in path:
        if "entity_id" in node:
            segments.append(node.get("label") or node["entity_id"])
        elif "relation_id" in node:
            segments.append(node.get("label") or node["relation_id"])
    return " → ".join(segments)
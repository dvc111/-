"""路径 → 自然语言文本，供 LLM 或用户阅读。"""

def verbalize(path):
    seg = []
    for node in path:
        if "entity_id" in node:
            seg.append(node.get("label") or node["entity_id"])
        elif "relation_id" in node:
            seg.append(node.get("label") or node["relation_id"])
    return " → ".join(seg)
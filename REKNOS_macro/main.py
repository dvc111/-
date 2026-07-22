# -*- coding: utf-8 -*-
"""
命令行入口。风格延续原 REKNOS 项目 main_freebase.py 的 argparse 用法。

用法示例：
    # 需先在本地启动 Ollama 并拉取 phi3： ollama run phi3
    python main.py --question "阿司匹林和布洛芬同时服用会有什么风险？" --question_id q_0001

    # 批量处理一个 JSON 文件（[{"question_id":..., "question_text":...}, ...]）
    python main.py --input questions.json --out result.json
"""

import argparse
import json
import uuid

import config
from kg_store import KGStore
from pipeline import macro_retrieval


def main():
    parser = argparse.ArgumentParser(description="REKNOS 宏观检索模块（实体链接 + 超关系评分 + 宏观子图裁剪）")
    parser.add_argument("--question", type=str, default=None, help="单条问题文本")
    parser.add_argument("--question_id", type=str, default=None, help="问题 ID，缺省自动生成")
    parser.add_argument("--input", type=str, default=None,
                         help="批量输入 JSON 文件，格式 [{'question_id':..,'question_text':..}, ...]")
    parser.add_argument("--out", type=str, default=None, help="结果输出 JSON 文件路径")
    parser.add_argument("--kg", type=str, default=config.DEFAULT_KG_PATH, help="本地 KG 文件路径")
    parser.add_argument("--LLM_type", type=str, default=config.LLM_TYPE, help="本地 LLM 名称（Ollama 模型名）")
    parser.add_argument("--width", type=int, default=config.TOP_K_HYPER_RELATIONS,
                         help="选取的高分超关系数量，对应 selected_hyper_relations 的规模")
    parser.add_argument("--max_hops", type=int, default=config.MAX_HOPS, help="宏观子图裁剪的最大搜索深度")
    args = parser.parse_args()

    kg = KGStore(args.kg)

    if args.input:
        with open(args.input, encoding="utf-8") as f:
            items = json.load(f)
    elif args.question:
        items = [{"question_id": args.question_id or f"q_{uuid.uuid4().hex[:8]}",
                   "question_text": args.question}]
    else:
        parser.error("必须提供 --question 或 --input 之一")
        return

    results = []
    for item in items:
        result = macro_retrieval(
            question_id=item["question_id"],
            question_text=item["question_text"],
            kg=kg,
            top_k_hyper_relations=args.width,
            max_hops=args.max_hops,
        )
        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results if len(results) > 1 else results[0], f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存至 {args.out}")


if __name__ == "__main__":
    main()

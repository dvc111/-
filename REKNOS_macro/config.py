# -*- coding: utf-8 -*-
"""
全局配置。
延续原 REKNOS 项目 main_freebase.py 中用 argparse 集中管理超参数的思路，
这里改为一个可被 argparse 覆盖的默认配置模块，方便脚本 / 单元测试共用。
"""

import os

# ---- LLM 相关（约束：使用 Phi-3，本地 Ollama 部署，不调用外部大模型 API） ----
LLM_TYPE = "phi3"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"          # Ollama 不校验 key，任意字符串即可
LLM_MAX_TOKENS = 512
LLM_TEMPERATURE = 0.2
LLM_MAX_RETRY = 3                  # 与原版 run_llm 的“死循环重试”不同，本地演示环境下重试有限次数

# ---- 知识图谱 ----
DEFAULT_KG_PATH = os.path.join(os.path.dirname(__file__), "kg", "toy_medical_kg.json")

# ---- 宏观检索超参数（对应原版 args.width / args.depth） ----
TOP_K_HYPER_RELATIONS = 2      # 每次选取的高分超关系数量，对应文档 selected_hyper_relations
MAX_HOPS = 2                   # 宏观子图裁剪的最大搜索深度，对应文档 max_hops
ENTITY_LINK_TOPK = 1           # 每个 mention 保留的候选实体个数
ENTITY_LINK_MIN_SIM = 0.35     # 实体链接的最低相似度阈值（低于此值判定为链接失败）

SCHEMA_VERSION = "0.1"

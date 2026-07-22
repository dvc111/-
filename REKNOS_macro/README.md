# REKNOS 宏观检索模块（软著一：宏观检索 + GNN 中的“宏观检索”部分）

本目录基于你上传的 `REKNOS.zip`（ICLR 2025《Reasoning of LLMs over Knowledge Graphs
with Super-Relations》官方实现）改写而成，实现《模块接口文档.md》第 1 节
`macro_subgraph` 接口所要求的三个子功能：

1. **实体链接**（`entity_linking.py`）
2. **超关系语义评分**（`hyper_relation_scoring.py`）
3. **宏观子图裁剪**（`macro_subgraph.py`）

不包含 GNN 节点重要性分析（对应文档第 2 节接口），那是软著一里"GNN"部分，按你的要求本次不实现。

## 与原始代码的关系（继承了什么方法思想）

| 原始代码 | 本模块 | 继承的思路 |
|---|---|---|
| `utils.py: run_llm()` | `llm_client.py: run_llm()` | 通过 OpenAI 兼容接口访问本地 Ollama，调用本地小模型；只是把无限重试改成有限重试+清晰报错，避免离线环境卡死 |
| `freebase_func.py: extract_relation_prompt` + `clean_relations()` | `prompt_list.py: HYPER_RELATION_SCORE_PROMPT` + `hyper_relation_scoring.py: _parse_scores()` | "LLM 用 few-shot 打分 -> 正则解析 `{标签 (Score: 0.x)}` -> 排序取 top-K" 的组合方法，只是打分对象从具体 relation 换成了超关系（super-relation），呼应论文核心概念 |
| `freebase_func.py: relation_search_prune_2hop()` | `macro_subgraph.py: prune_macro_subgraph()` | 从主题实体出发做多跳搜索；原版用 SPARQL 查 Freebase，这里在本地 KG 上做 BFS，并用"已选超关系"约束扩展方向（对应文档"依据高分超关系约束图遍历方向，实现知识领域快速定位"） |
| `freebase_func.py: entity_prune()` | 同上（`prune_macro_subgraph` 内的裁剪 + 排序） | 按分数保留候选、剪掉不相关分支的思路 |
| `utils.py: clean_scores()` 打分解析失败时的均匀分配兜底 | `hyper_relation_scoring.py` 中 `if not parsed: ... 均匀打分` | 保留原版"解析失败不中断流程"的稳健性设计 |

## 满足的约束

- **不使用 Freebase / Wikidata**：`kg_store.py` 直接读取本地小型 JSON 文件
  `kg/toy_medical_kg.json`（自建医药领域示例 KG，约 15 个实体 / 9 个关系 / 5 个超关系 / 22 条三元组），
  不再调用原版的 SPARQL / Freebase 接口。
- **不下载大型 KG**：KG 文件随代码提供，几 KB 大小。
- **LLM 使用 Phi-3**：`config.py` 中 `LLM_TYPE = "phi3"`，通过本地 Ollama（`http://localhost:11434/v1`）调用，
  与原版 `main_freebase.py --LLM_type phi3` 的用法一致。
- **只实现宏观检索三步**：实体链接 / 超关系语义评分 / 宏观子图裁剪；不含 GNN。

## 输出格式

`pipeline.macro_retrieval()` 的返回值与《模块接口文档.md》第 1 节 `macro_subgraph` 接口
完全对齐：`schema_version` / `question_id` / `question_text` / `entity_linking` /
`topic_entities` / `selected_hyper_relations` / `macro_subgraph.nodes` /
`macro_subgraph.triples` / `max_hops`，可直接作为 GNN 推理模块（软著一另一部分，未来补齐）的输入。

`initial_importance` 的计算方式在 `macro_subgraph.py: _initial_importance()` 中，
按文档字段说明"由实体链接置信度、超关系匹配程度及节点中心性综合计算"：
- 主题实体：`0.7 * 实体链接置信度 + 0.3 * 归一化度数`
- 子图中其它实体：`0.6 * 引入该节点的最高超关系得分 + 0.4 * 归一化度数`

## 运行方式

### 1. 不依赖真实 LLM 的离线结构测试（推荐先跑这个）

```bash
cd REKNOS_macro
python test_pipeline.py
```

用一个确定性的 mock LLM 函数验证整条流水线（实体链接→超关系评分→子图裁剪）
产出的 JSON 结构、字段、取值范围是否符合接口文档，不需要启动 Ollama。

### 2. 接入真实 Phi-3（需要本机已安装并启动 Ollama）

```bash
ollama pull phi3
ollama serve            # 若尚未在后台运行

cd REKNOS_macro
pip install -r requirements.txt
python main.py --question "阿司匹林和布洛芬同时服用会有什么风险？" --question_id q_0001 --out result.json
```

也可以批量跑：

```bash
python main.py --input questions.json --out results.json
```

`questions.json` 格式：`[{"question_id": "q_0001", "question_text": "..."}, ...]`

## 目录结构

```
REKNOS_macro/
├── kg/
│   └── toy_medical_kg.json     # 自建本地小型 KG（医药领域示例）
├── config.py                   # 全局配置（LLM 类型、超参数、路径）
├── prompt_list.py               # 提示词模板（延续原版 few-shot + 结构化输出的风格）
├── llm_client.py                # 本地 Phi-3 (Ollama) 调用封装
├── kg_store.py                  # 本地 KG 加载与索引（替代原版 freebase_func.py 的 SPARQL 层）
├── entity_linking.py            # 实体链接
├── hyper_relation_scoring.py    # 超关系语义评分
├── macro_subgraph.py            # 宏观子图裁剪
├── pipeline.py                  # 三步组装，产出 macro_subgraph 接口 JSON
├── main.py                      # CLI 入口（风格延续原版 main_freebase.py）
├── test_pipeline.py             # 离线 mock 测试
└── requirements.txt
```

## 后续可扩展点（若要继续实现软著一的 GNN 部分）

`pipeline.macro_retrieval()` 的输出可直接喂给一个 GNN 模块，按《模块接口文档.md》
第 2 节 `gnn_reasoning_output` 接口，在 `macro_subgraph.nodes/triples` 上做消息传播，
输出 `node_importance` 与 `candidate_answers`，与本模块的字段命名、id 体系完全兼容。

# 微观检索模块接口 v0.1

## 模块边界

```text
宏观模块 macro_subgraph
        ↓
微观模块 DDE + 语义特征 + MLP/轻量评分 + Top-K
        ↓
GNN 模块 micro_evidence_subgraph
```

微观模块不重新读取全量知识图谱，也不负责实体链接、宏观 BFS、R-GCN、答案节点预测或 LLM 生成。

## HTTP 接口

```text
POST /api/v1/micro-evidence-subgraph?top_k=20&threshold=0&require_mlp=false
Content-Type: application/json
```

请求体示例见 `examples/macro_subgraph_v0.1.json`。

## 必需输入字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `schema_version` | string | 当前固定为 `0.1` |
| `question_id` | string | 一次问答的唯一编号 |
| `question_text` | string | 原始问题文本 |
| `topic_entities` | string[] | 已完成实体链接的主题实体 ID |
| `macro_subgraph.nodes` | object[] | 实体 ID、标签和主题标记 |
| `macro_subgraph.triples` | string[][] | `[head_id, relation_id, tail_id]` |

`relation_labels` 是推荐的可选字段，格式为 `{relation_id: label}`。如果上游未提供关系标签，微观模块会使用关系 ID 作为文本，不会导致接口失败，但语义评分质量可能降低。

## 输出字段

| 字段 | 类型 | 含义 |
| --- | --- | --- |
| `schema_version` | string | 固定为 `0.1` |
| `question_id` | string | 原样返回，方便全链路追踪 |
| `evidence_triples` | object[] | 按相关度降序排列的 Top-K 证据 |
| `relevance_score` | float | `[0,1]` 范围内的证据相关度 |
| `dde.head_dde` | float[] | 头实体的有向距离编码 |
| `dde.tail_dde` | float[] | 尾实体的有向距离编码 |
| `node_features` | object[] | 节点名称和可直接转 Tensor 的文本向量；有真实共享索引时也保留索引 |
| `entity_dde` | object | 实体 ID → DDE，方便 GNN 直接构造张量字典 |
| `relation_labels` | object | 证据关系 ID → 自然语言名称，供路径线性化使用 |
| `relation_to_id` | object | 本次证据子图使用的固定关系编号 |
| `feature_spec` | object | DDE、文本嵌入和 GNN 输入维度约定 |
| `scoring` | object | 当前评分器类型、模型加载状态和阈值 |

## DDE 约定

当前 `rounds=2`，实体 DDE 的固定拼接顺序为：

```text
[主题实体二值 one-hot,
 第1轮入边传播, 第2轮入边传播,
 第1轮出边传播, 第2轮出边传播]
```

每部分为 2 维，因此总长度为 `2 + 2×2 + 2×2 = 10`。GNN 输入层需要按 10 维配置。若双方以后修改传播轮数，必须同时更新微观模块和 GNN 配置。

当前原型文本嵌入为 128 维，因此：

```text
GNN in_dim = text_embedding_dim + dde_dim = 128 + 10 = 138
```

接口会在 `feature_spec.gnn_input_dim` 中直接返回该值。

每个 DDE 块为 2 维，接口同时返回 `feature_spec.dde_block_dim=2`。

## 文本嵌入约定

为避免输出无法加载的假索引，微观模块始终在 `node_features.text_embedding` 中提供真实的 128 维数组。GNN 可以直接转换：

```python
node_embeddings = {
    item["entity_id"]: torch.tensor(item["text_embedding"], dtype=torch.float32)
    for item in payload["node_features"]
}

entity_dde = {
    entity_id: torch.tensor(vector, dtype=torch.float32)
    for entity_id, vector in payload["entity_dde"].items()
}
```

也可以直接使用经过完整校验的适配器：

```python
from micro_rag import prepare_gnn_handoff

handoff = prepare_gnn_handoff(payload)
# handoff.triples
# handoff.node_embeddings
# handoff.entity_dde
# handoff.relation_to_id
# handoff.gnn_input_dim
```

共享向量库上线后，上游可在 node 中提供真实 `text_embedding_id`，微观模块会原样传递，同时保留内联向量作为回退。

## 评分器运行模式

默认无模型时使用可解释规则评分，并明确返回：

```json
{"scorer_type": "heuristic", "model_loaded": false}
```

加载训练模型后会返回 `scorer_type: "mlp"`。正式联调应启用 `require_mlp=true`；未加载模型时程序会直接报错，避免把规则分数误称为 MLP 结果。

## Python 直接调用

```python
import json
from micro_rag.contracts import build_micro_evidence_subgraph

with open("examples/macro_subgraph_v0.1.json", encoding="utf-8") as source:
    macro_result = json.load(source)

micro_result = build_micro_evidence_subgraph(macro_result, top_k=20)
```

## 联调检查项

- 三元组引用的头尾实体必须存在于 `nodes`。
- `topic_entities` 使用实体 ID，不使用显示名称。
- `evidence_triples` 必须按 `relevance_score` 降序排列。
- 所有 `head_dde`、`tail_dde` 长度必须一致。
- `node_features` 只包含 Top-K 证据子图实际涉及的实体。
- `feature_spec.gnn_input_dim` 应与 R-GCN 初始化的 `in_dim` 一致。
- 正式实验应确认 `scoring.scorer_type` 为 `mlp`。
- 不支持的 `schema_version` 会返回 400，避免静默误读字段。

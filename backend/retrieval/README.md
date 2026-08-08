# SubgraphRAG 微观检索模块（轻量版）



## 实现内容

流程为：

1. 接收问题、主题实体和宏观局部子图。
2. 沿入边与出边分别进行多轮均值传播，形成实体级 DDE。
3. 计算问题与实体、关系、三元组的轻量语义相关度。
4. 拼接语义特征与三元组两端实体的 DDE。
5. 使用默认可解释评分器，或使用最短路径弱监督训练的两层 MLP，并行计算候选分数。
6. 排序并输出 Top-K 微观证据子图及各项分数。
7. 通过统一适配器校验并拆分 GNN 所需节点向量、DDE 和关系映射。

论文原版使用 GTE 文本编码器和 PyTorch MLP。本实现为了让普通笔记本直接运行，默认采用确定性哈希文本编码器和纯 Python MLP，不需要显卡。后续只需替换 `HashingTextEncoder`，其余接口不变。

## 团队联调接口 v0.1

正式联调输入采用实体 ID 和关系 ID。宏观模块传入：

```json
{
  "schema_version": "0.1",
  "question_id": "q_0001",
  "question_text": "阿司匹林和布洛芬同时服用会有什么风险？",
  "topic_entities": ["Q1024", "Q1088"],
  "relation_labels": {"R017": "增加风险"},
  "macro_subgraph": {
    "nodes": [
      {"entity_id": "Q1024", "label": "阿司匹林", "is_topic": true},
      {"entity_id": "Q2031", "label": "胃肠道出血", "is_topic": false}
    ],
    "triples": [["Q1024", "R017", "Q2031"]]
  }
}
```

输出为 GNN 组约定的 `micro_evidence_subgraph`：

```json
{
  "schema_version": "0.1",
  "question_id": "q_0001",
  "evidence_triples": [
    {
      "triple": ["Q1024", "R017", "Q2031"],
      "relevance_score": 0.88,
      "dde": {"head_dde": [], "tail_dde": []}
    }
  ],
  "top_k": 20,
  "node_features": [
    {
      "entity_id": "Q1024",
      "label": "阿司匹林",
      "text_embedding": [0.0, 0.1]
    }
  ],
  "entity_dde": {"Q1024": [0.0, 1.0]},
  "relation_labels": {"R017": "增加风险"},
  "feature_spec": {
    "dde_dim": 10,
    "text_embedding_dim": 128,
    "gnn_input_dim": 138
  },
  "scoring": {"scorer_type": "heuristic", "model_loaded": false}
}
```

完整字段说明见 [INTERFACE_V0.1.md](INTERFACE_V0.1.md)。旧的名称型输入仍然兼容，供单机演示和训练使用。

原型接口直接携带真实的 128 维文本向量，GNN 可直接转成 Tensor。只有上游提供真实共享库索引时才会附带 `text_embedding_id`，不会生成不存在的 `emb_Q...`。

## 直接运行演示

在本目录执行：

```powershell
python -m micro_rag.cli retrieve --input examples/macro_subgraph_v0.1.json --top-k 20
```

核心模块只依赖 Python 标准库。

还可以分别测试人物和作品领域样例：

```powershell
python -m micro_rag.cli retrieve --input examples/einstein_macro_subgraph_v0.1.json --top-k 2
python -m micro_rag.cli retrieve --input examples/book_macro_subgraph_v0.1.json --top-k 2
```

## 启动 HTTP 后端

```powershell
python -m pip install -e ".[api]"
python -m uvicorn micro_rag.api:app --host 0.0.0.0 --port 8000 --reload
```

接口：

- `GET /health`
- `POST /api/v1/micro-evidence-subgraph?top_k=20&threshold=0&require_mlp=false`
- `POST /api/v1/micro-retrieve`（旧版兼容接口）

联调时把宏观模块的完整 JSON 作为请求体发送给 `micro-evidence-subgraph`。

## GNN 交接适配器

队友仓库的 GNN 需要三元组、实体向量、DDE 和固定关系编号。现在可以集中转换，不必在服务端和演示脚本中重复手写解析：

```python
from micro_rag import prepare_gnn_handoff

handoff = prepare_gnn_handoff(micro_result)
print(handoff.gnn_input_dim)       # 138
print(handoff.relation_to_id)      # 例如 {"R017": 0}
```

安装 PyTorch 后也可直接得到张量：

```powershell
python -m pip install -e ".[gnn]"
```

```python
from micro_rag import prepare_torch_gnn_inputs
gnn_inputs = prepare_torch_gnn_inputs(micro_result)
```

## 训练轻量 MLP

训练数据采用 JSONL，每行在检索输入基础上增加 `answer_entities`。程序会自动寻找主题实体到答案实体的最短路径，将路径三元组标为正样本，其余标为负样本：

```powershell
python -m micro_rag.cli train --data examples/train.jsonl --output model.json --epochs 300
python -m micro_rag.cli retrieve --input examples/macro_subgraph_v0.1.json --model model.json --top-k 20 --require-mlp
```

部署接口时可设置环境变量 `MICRO_RAG_MODEL` 为模型文件路径。

## 运行测试

```powershell
python -m unittest discover -s tests -v
```


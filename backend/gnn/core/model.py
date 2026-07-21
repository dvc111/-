"""
R-GCN 共享模型

宏观和微观 GNN 模块共用的关系图卷积网络（Relational Graph Convolutional Network）。
支持可配置的层数、隐藏维度和关系数量，两端共享同一基类，
通过不同的输入特征拼接和下游头部实现各自的任务。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from dataclasses import dataclass


class RGCNLayer(nn.Module):
    """单层关系图卷积。

    对每个节点，按关系类型分别聚合邻居信息后取和，
    与自身变换后的表示相加，经激活后输出。
    """

    def __init__(self, in_dim: int, out_dim: int, num_relations: int,
                 self_loop: bool = True, dropout: float = 0.0,
                 activation=F.relu, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.self_loop = self_loop
        self.activation = activation

        # 每种关系类型对应一个线性变换 W_r
        self.relation_weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
            for _ in range(num_relations)
        ])

        # 自环变换 W_self
        if self_loop:
            self.loop_weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        else:
            self.register_parameter('loop_weight', None)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """前向传播。

        Args:
            x:          节点特征矩阵, shape (N, in_dim)
            edge_index:  边索引, shape (2, E)
            edge_type:   边类型, shape (E,), 取值 [0, num_relations-1]

        Returns:
            更新后的节点表示, shape (N, out_dim)
        """
        device = x.device
        N = x.size(0)
        out = torch.zeros(N, self.out_dim, device=device)

        # 按关系类型分别聚合
        for rel in range(self.num_relations):
            mask = edge_type == rel
            if not mask.any():
                continue
            src = edge_index[0, mask]
            dst = edge_index[1, mask]
            # (E_rel, out_dim) = (E_rel, in_dim) @ (in_dim, out_dim)
            neighbor_msg = x[src] @ self.relation_weights[rel]
            out.index_add_(0, dst, neighbor_msg)

        # 自环
        if self.self_loop:
            out = out + x @ self.loop_weight

        if self.bias is not None:
            out = out + self.bias

        out = self.dropout(out)
        if self.activation is not None:
            out = self.activation(out)
        return out


class RGCN(nn.Module):
    """通用 R-GCN 模型，宏观 / 微观模块共用。

    两层关系图卷积，中间加 ReLU + Dropout，输出节点嵌入。
    两端通过不同的下游头（分类器 / 评分器）实现各自任务。
    """

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_relations: int, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.num_layers = num_layers

        layers = []
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        for i in range(num_layers):
            act = F.relu if i < num_layers - 1 else None
            layers.append(RGCNLayer(
                in_dim=dims[i],
                out_dim=dims[i + 1],
                num_relations=num_relations,
                self_loop=True,
                dropout=dropout if i < num_layers - 1 else 0.0,
                activation=act,
                bias=True,
            ))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """返回最后一层的节点嵌入。

        Args:
            x:          节点特征, shape (N, in_dim)
            edge_index:  边索引, shape (2, E)
            edge_type:   边类型, shape (E,)

        Returns:
            节点嵌入, shape (N, out_dim)
        """
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        return x


class RGCNNodeClassifier(nn.Module):
    """R-GCN + 节点分类头：宏观和微观共用。

    先经 RGCN 编码得到节点嵌入，再通过线性层 + softmax 映射为概率分布。
    宏观侧用于子图节点重要性二分类（重要 / 非重要），
    微观侧用于候选答案多分类或二分类评分。
    """

    def __init__(self, in_dim: int, hidden_dim: int,
                 num_relations: int, num_classes: int = 2,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = RGCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_relations=num_relations,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """返回节点级分类 logits, shape (N, num_classes)。"""
        node_emb = self.encoder(x, edge_index, edge_type)
        return self.classifier(node_emb)

    def predict_proba(self, x: torch.Tensor, edge_index: torch.Tensor,
                      edge_type: torch.Tensor) -> torch.Tensor:
        """返回 softmax 归一化后的节点概率, shape (N, num_classes)。"""
        logits = self.forward(x, edge_index, edge_type)
        return F.softmax(logits, dim=-1)


class RGCNNodeScorer(nn.Module):
    """R-GCN + 单节点评分头：用于候选答案概率估计。

    GNN 推理模块中，将节点嵌入映射为 [0,1] 区间内的答案概率。
    宏微观两侧的 scoring 层通用。
    """

    def __init__(self, in_dim: int, hidden_dim: int,
                 num_relations: int, num_layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.encoder = RGCN(
            in_dim=in_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            num_relations=num_relations,
            num_layers=num_layers,
            dropout=dropout,
        )
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """返回节点级评分, shape (N, 1)，取值 [0,1]。"""
        node_emb = self.encoder(x, edge_index, edge_type)
        return self.scorer(node_emb)


@dataclass
class InferenceResult:
    """统一推理结果。

    仅包含模型原始输出，不含任何业务层解释（无 Top-K、无路径、无答案）。
    """
    node_embeddings: torch.Tensor   # (N, out_dim)
    predictions: Optional[torch.Tensor]  # (N, C) 分类 logits 或 (N, 1) 评分


def run_inference(
    model: nn.Module,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    edge_type: torch.Tensor,
    device: Optional[torch.device] = None,
) -> InferenceResult:
    """统一推理入口：设备放置 → eval 模式 → 无梯度前向 → InferenceResult。

    纯机械流程，不含任何业务逻辑。
    支持 RGCN / RGCNNodeClassifier / RGCNNodeScorer 三种模型。
    调用方自行决定是否传入 device；不传则沿用输入张量所在设备。

    Args:
        model:      实例化的 RGCN / RGCNNodeClassifier / RGCNNodeScorer
        x:          节点特征, shape (N, in_dim)
        edge_index:  边索引, shape (2, E)
        edge_type:   边类型, shape (E,)
        device:     可选，目标推理设备

    Returns:
        InferenceResult:
          - node_embeddings: RGCN 编码器最后一层的节点嵌入 (N, out_dim)
          - predictions:     下游头输出；纯编码器模型则为 None
    """
    target_device = device if device is not None else x.device

    model.eval()
    model.to(target_device)

    x = x.to(target_device)
    edge_index = edge_index.to(target_device)
    edge_type = edge_type.to(target_device)

    with torch.no_grad():
        if isinstance(model, RGCN):
            node_emb = model(x, edge_index, edge_type)
            return InferenceResult(node_embeddings=node_emb, predictions=None)

        if isinstance(model, RGCNNodeClassifier):
            node_emb = model.encoder(x, edge_index, edge_type)
            logits = model.classifier(node_emb)
            return InferenceResult(node_embeddings=node_emb, predictions=logits)

        if isinstance(model, RGCNNodeScorer):
            node_emb = model.encoder(x, edge_index, edge_type)
            scores = model.scorer(node_emb)
            return InferenceResult(node_embeddings=node_emb, predictions=scores)

        # 兜底：未知模型类型，作为纯编码器执行
        node_emb = model(x, edge_index, edge_type)
        return InferenceResult(node_embeddings=node_emb, predictions=None)

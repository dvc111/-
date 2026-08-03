import torch
from typing import Optional
from dataclasses import dataclass
import torch.nn as nn
import torch.nn.functional as F

class RGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, num_relations: int,
                 self_loop: bool = True, dropout: float = 0.0,
                 activation=F.relu, bias: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_relations = num_relations
        self.self_loop = self_loop
        self.activation = activation
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.relation_weights = nn.ParameterList([
            nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
            for _ in range(num_relations)
        ])
        if self_loop:
            self.loop_weight = nn.Parameter(torch.randn(in_dim, out_dim) * 0.1)
        else:
            self.register_parameter('loop_weight', None)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_dim))
        else:
            self.register_parameter('bias', None)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x, edge_index, edge_type, trace=False)[0]

    def forward_with_trace(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor):
        """R-GCN layer forward that also exposes per-relation messages for visualization."""
        return self._forward_impl(x, edge_index, edge_type, trace=True)

    def _forward_impl(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor, trace: bool):
        device = x.device
        N = x.size(0)
        out = torch.zeros(N, self.out_dim, device=device)
        messages = []
        for rel in range(self.num_relations):
            mask = edge_type == rel
            if not mask.any():
                continue
            src, dst = edge_index[0, mask], edge_index[1, mask]
            neighbor_msg = x[src] @ self.relation_weights[rel]
            out.index_add_(0, dst, neighbor_msg)
            messages.append({"relation": rel, "src": src, "dst": dst, "message": neighbor_msg})
        self_loop = None
        if self.self_loop:
            self_loop = x @ self.loop_weight
            out = out + self_loop
        if self.bias is not None:
            out = out + self.bias
        out = self.dropout(out)
        if self.activation is not None:
            out = self.activation(out)
        if trace:
            return out, {"messages": messages, "self_loop": self_loop}
        return out, None


class RGCN(nn.Module):
    """通用 R-GCN 编码器，默认 2 层。宏观/微观共用编码部分。"""
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int,
                 num_relations: int, num_layers: int = 2, dropout: float = 0.2):
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
            layers.append(RGCNLayer(in_dim=dims[i], out_dim=dims[i + 1], num_relations=num_relations,
                                    self_loop=True, dropout=dropout if i < num_layers - 1 else 0.0, activation=act, bias=True))
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        return x


class RGCNNodeClassifier(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_relations: int,
                 num_classes: int = 2, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.encoder = RGCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
                            num_relations=num_relations, num_layers=num_layers, dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, x, edge_index, edge_type):
        return self.classifier(self.encoder(x, edge_index, edge_type))

    def predict_proba(self, x, edge_index, edge_type):
        return F.softmax(self.forward(x, edge_index, edge_type), dim=-1)


class RGCNNodeScorer(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, num_relations: int,
                 num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_relations = num_relations
        self.encoder = RGCN(in_dim=in_dim, hidden_dim=hidden_dim, out_dim=hidden_dim,
                            num_relations=num_relations, num_layers=num_layers, dropout=dropout)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1), nn.Sigmoid())

    def forward(self, x, edge_index, edge_type):
        return self.scorer(self.encoder(x, edge_index, edge_type))


# ── 统一推理入口 ──
@dataclass
class InferenceResult:
    node_embeddings: torch.Tensor  # (N, out_dim)
    predictions: Optional[torch.Tensor]  # (N, C) logits 或 (N, 1) 分数


def run_inference(model, x, edge_index, edge_type, device=None):
    """设备分配 → eval 模式 → 无梯度前向 → 返回嵌入和预测结果。"""
    target_device = device if device is not None else x.device
    model.eval()
    model.to(target_device)
    x = x.to(target_device)
    edge_index = edge_index.to(target_device)
    edge_type = edge_type.to(target_device)
    with torch.no_grad():
        if isinstance(model, RGCN):
            return InferenceResult(node_embeddings=model(x, edge_index, edge_type), predictions=None)
        if isinstance(model, RGCNNodeClassifier):
            emb = model.encoder(x, edge_index, edge_type)
            return InferenceResult(node_embeddings=emb, predictions=model.classifier(emb))
        if isinstance(model, RGCNNodeScorer):
            emb = model.encoder(x, edge_index, edge_type)
            return InferenceResult(node_embeddings=emb, predictions=model.scorer(emb))
        return InferenceResult(node_embeddings=model(x, edge_index, edge_type), predictions=None)

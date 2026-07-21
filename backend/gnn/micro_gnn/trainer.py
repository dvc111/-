"""微观模型训练。

加载微观训练数据集（含 DDE 标注），训练 RGCNNodeScorer（概率评分），
保存 micro_model.pth。微观 GNN 的权重必须用含 DDE 的微观子图训练出来。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

_BACKEND_DIR = str(Path(__file__).resolve().parents[2])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from gnn.core.model import RGCNNodeScorer
from gnn.core.base_loader import GraphData

CHECKPOINT_CONFIG_KEY = "_model_config"


def _run_one_epoch(
    model: nn.Module,
    data: list[tuple[GraphData, torch.Tensor]],
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> float:
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss = 0.0
    total_nodes = 0

    with torch.set_grad_enabled(is_train):
        for graph_data, scores in data:
            x = graph_data.node_features.to(device)
            ei = graph_data.edge_index.to(device)
            et = graph_data.edge_type.to(device)
            target = scores.to(device).view(-1, 1)

            pred = model(x, ei, et)
            loss = nn.functional.binary_cross_entropy(pred, target)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * target.size(0)
            total_nodes += target.size(0)

    return total_loss / total_nodes


def train_micro_model(
    train_data: list[tuple[GraphData, torch.Tensor]],
    val_data: Optional[list[tuple[GraphData, torch.Tensor]]] = None,
    in_dim: int = 832,
    hidden_dim: int = 128,
    num_relations: int = 10,
    num_layers: int = 2,
    dropout: float = 0.2,
    lr: float = 0.001,
    epochs: int = 50,
    save_path: str = "micro_model.pth",
    device: Optional[torch.device] = None,
    print_every: int = 10,
) -> RGCNNodeScorer:
    """训练微观 RGCNNodeScorer（概率评分模型）。

    Args:
        train_data:    训练集，每项 (GraphData, scores)，scores 为 (N,) 的 [0,1] 目标概率。
        val_data:      验证集（可选），格式同 train_data。
        in_dim:        输入维度。默认 832 = BERT(768) + DDE(64)。
        hidden_dim:    隐藏层维度。
        num_relations: 关系类型总数。
        num_layers:    RGCN 层数。
        dropout:       Dropout 比率。
        lr:            学习率。
        epochs:        训练轮数。
        save_path:     模型保存路径，默认 micro_model.pth。
        device:        训练设备。
        print_every:   每 N 轮打印一次 loss。

    Returns:
        训练好的 RGCNNodeScorer 实例。
    """
    target_device = device or (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )

    model = RGCNNodeScorer(
        in_dim=in_dim,
        hidden_dim=hidden_dim,
        num_relations=num_relations,
        num_layers=num_layers,
        dropout=dropout,
    ).to(target_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        train_loss = _run_one_epoch(model, train_data, optimizer, target_device)

        val_loss: Optional[float] = None
        if val_data:
            val_loss = _run_one_epoch(model, val_data, None, target_device)

        if epoch == 1 or epoch % print_every == 0 or epoch == epochs:
            msg = f"[{epoch:3d}/{epochs}]  train_loss={train_loss:.4f}"
            if val_loss is not None:
                msg += f"  val_loss={val_loss:.4f}"
            print(msg)

    checkpoint = {
        CHECKPOINT_CONFIG_KEY: {
            "in_dim": in_dim,
            "hidden_dim": hidden_dim,
            "num_relations": num_relations,
            "num_layers": num_layers,
            "dropout": dropout,
        },
    }
    checkpoint.update(model.state_dict())
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")

    return model
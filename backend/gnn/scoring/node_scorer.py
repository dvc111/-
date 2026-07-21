"""
节点评分器：加载模型 + 前向传播，一步完成。

接收 GraphData 和模型权重路径，执行 run_inference，
返回节点概率向量。宏观和微观的 inference.py 都会调用此模块。
"""

from __future__ import annotations

import sys
from pathlib import Path
import torch
from typing import Optional, Dict, Any

_BACKEND_DIR = str(Path(__file__).resolve().parents[2])
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

from gnn.core.model import RGCNNodeScorer, run_inference
from gnn.core.base_loader import GraphData

CHECKPOINT_CONFIG_KEY = "_model_config"


def _load_checkpoint(checkpoint_path: str, device: torch.device):
    raw = torch.load(checkpoint_path, map_location=device, weights_only=True)
    if isinstance(raw, dict) and CHECKPOINT_CONFIG_KEY in raw:
        return raw.pop(CHECKPOINT_CONFIG_KEY), raw
    if isinstance(raw, dict) and "model_config" in raw:
        return raw.pop("model_config"), raw
    if isinstance(raw, dict) and "state_dict" in raw:
        return raw.get("model_config", {}), raw["state_dict"]
    return {}, raw


def _build_model(model_config, graph_data, **kwargs):
    in_dim = model_config.get("in_dim") or kwargs.get("in_dim") or graph_data.node_features.size(-1)
    hidden_dim = model_config.get("hidden_dim") or kwargs.get("hidden_dim", 64)
    num_relations = model_config.get("num_relations") or kwargs.get("num_relations") or graph_data.num_relations
    num_layers = model_config.get("num_layers") or kwargs.get("num_layers", 2)
    dropout = model_config.get("dropout") or kwargs.get("dropout", 0.2)
    return RGCNNodeScorer(in_dim=in_dim, hidden_dim=hidden_dim, num_relations=num_relations, num_layers=num_layers, dropout=dropout)


def load_and_score(checkpoint_path: str, graph_data: GraphData, device: Optional[torch.device] = None, **kwargs) -> torch.Tensor:
    target_device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    model_config, state_dict = _load_checkpoint(checkpoint_path, target_device)
    model = _build_model(model_config, graph_data, **kwargs)
    model.load_state_dict(state_dict, strict=False)
    result = run_inference(model=model, x=graph_data.node_features, edge_index=graph_data.edge_index, edge_type=graph_data.edge_type, device=target_device)
    assert result.predictions is not None
    return result.predictions.view(-1)
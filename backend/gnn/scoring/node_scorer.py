"""加载模型 → 推理 → 节点概率向量。"""
import torch
from gnn.core.model import RGCNNodeScorer, RGCNNodeClassifier, run_inference
from gnn.core.base_loader import GraphData

def load_and_score(model, graph_data: GraphData, device=None) -> torch.Tensor:
    target_device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
    result = run_inference(model=model, x=graph_data.node_features, edge_index=graph_data.edge_index, edge_type=graph_data.edge_type, device=target_device)
    preds = result.predictions
    assert preds is not None
    if preds.dim() == 2 and preds.size(1) > 1:
        return torch.softmax(preds, dim=-1)[:, 1]
    return preds.view(-1)
"""轻量级 SubgraphRAG 微观检索模块。"""

from .models import Evidence, RetrievalResult, Triple
from .retriever import MicroRetriever
from .contracts import build_micro_evidence_subgraph, parse_macro_subgraph
from .gnn_adapter import GNNHandoff, prepare_gnn_handoff, prepare_torch_gnn_inputs

__all__ = [
    "Evidence",
    "MicroRetriever",
    "RetrievalResult",
    "Triple",
    "build_micro_evidence_subgraph",
    "parse_macro_subgraph",
    "GNNHandoff",
    "prepare_gnn_handoff",
    "prepare_torch_gnn_inputs",
]

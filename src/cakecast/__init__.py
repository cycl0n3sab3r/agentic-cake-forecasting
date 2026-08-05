"""cakecast — forecasting autonomous completion of a long task chain."""

from .evidence import (
    EvidenceError,
    Observation,
    load_evidence,
    pool_by_node,
    posterior_mean,
)
from .graph import GraphError, Step, Task, TaskGraph, load_graph
from .model import ClosureResult, chain_closure

__version__ = "0.1.0"

__all__ = [
    "ClosureResult",
    "EvidenceError",
    "GraphError",
    "Observation",
    "Step",
    "Task",
    "TaskGraph",
    "chain_closure",
    "load_evidence",
    "load_graph",
    "pool_by_node",
    "posterior_mean",
]

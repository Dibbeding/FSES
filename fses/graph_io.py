"""Helpers for loading hypergraphs and writing samples."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, MutableMapping, MutableSet, Tuple

EdgeList = Dict[int, MutableSet[int]]
NodeList = Dict[int, MutableSet[int]]
NodeCommunities = Dict[int, int]


def _parse_edge_line(raw_line: str) -> Iterable[int]:
    raw_line = raw_line.strip()
    if not raw_line:
        return []
    # Accept comma or whitespace separation
    if "," in raw_line:
        parts = raw_line.split(",")
    else:
        parts = raw_line.split()
    return [int(p) for p in parts if p.strip()]


def load_edge_list(path: Path) -> EdgeList:
    """Load hyperedges from a file into a dict: edge_id -> set[node]."""

    edges: EdgeList = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle):
            members = set(_parse_edge_line(raw_line))
            if not members:
                continue
            edges[idx] = members
    return edges


def build_node_list(edge_list: EdgeList) -> NodeList:
    """Invert edge list into node -> edges adjacency."""

    nodes: NodeList = {}
    for edge_id, members in edge_list.items():
        for node in members:
            nodes.setdefault(node, set()).add(edge_id)
    return nodes


def load_assignments(path: Path, present_nodes: Iterable[int] | None = None) -> NodeCommunities:
    """Load per-node community ids, optionally filtered to present_nodes."""
    keep = set(present_nodes) if present_nodes is not None else None
    assignments: NodeCommunities = {}
    with path.open("r", encoding="utf-8") as handle:
        for idx, raw_line in enumerate(handle, start=1):
            if keep is not None and idx not in keep:
                continue
            line = raw_line.strip()
            if not line:
                continue
            try:
                assignments[idx] = int(line)
            except Exception:
                assignments[idx] = 1
    return assignments


def load_hypergraph(
    hyperedge_path: Path,
    assign_path: Path,
) -> Tuple[EdgeList, NodeList, NodeCommunities]:
    """Load hyperedges + assignments and return edge list, node list, labels."""

    edges = load_edge_list(hyperedge_path)
    nodes = build_node_list(edges)
    assignments = load_assignments(assign_path, present_nodes=nodes.keys())
    return edges, nodes, assignments

"""
Joint Unbiased Random Walk (JURW) baseline with Metropolis-Hastings normalization.

Self-contained implementation based on the description in
“Sampling hypergraphs via joint unbiased random walk”. No external repo required.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, Mapping, MutableSet

EdgeId = int
NodeId = int
EdgeList = Mapping[EdgeId, Iterable[NodeId]]
NodeList = Mapping[NodeId, Iterable[EdgeId]]


def _build_neighbor_counts(
    node: NodeId,
    edge_list: EdgeList,
    node_list: NodeList,
) -> tuple[Dict[NodeId, int], int]:
    """Count how often each neighbor co-occurs with `node` across its incident edges."""
    counts: Dict[NodeId, int] = {}
    total = 0
    for eid in node_list[node]:
        members = edge_list[eid]
        for neigh in members:
            if neigh == node:
                continue
            counts[neigh] = counts.get(neigh, 0) + 1
            total += 1
    return counts, total


def expansion_sampling_JURW_MHNorm(
    dict_edge_list: EdgeList,
    dict_node_list: NodeList,
    target_node_count: int,
) -> Dict[str, object]:
    """Run a JURW walk with MH-normalized transitions until the node budget is met."""

    # Cache of neighbor counts per node to avoid recomputation
    neighbor_cache: Dict[NodeId, tuple[Dict[NodeId, int], int]] = {}

    sampled_edges: MutableSet[EdgeId] = set()
    sampled_nodes: MutableSet[NodeId] = set()

    nodes = list(dict_node_list.keys())
    current_node = random.choice(nodes)
    incident_edges = list(dict_node_list[current_node])
    current_edge = random.choice(incident_edges)
    sampled_edges.add(current_edge)
    sampled_nodes.update(dict_edge_list[current_edge])

    while len(sampled_nodes) < target_node_count:
        if current_node not in neighbor_cache:
            neighbor_cache[current_node] = _build_neighbor_counts(current_node, dict_edge_list, dict_node_list)
        neigh_counts, total_neigh = neighbor_cache[current_node]

        if not neigh_counts or total_neigh == 0:
            # Dead-end: restart from a random node
            current_node = random.choice(nodes)
            incident_edges = list(dict_node_list[current_node])
            current_edge = random.choice(incident_edges)
            sampled_edges.add(current_edge)
            sampled_nodes.update(dict_edge_list[current_edge])
            continue

        # Choose next node with MH-normalized weights
        neighs = []
        weights = []
        deg_current = max(1, len(dict_node_list[current_node]))
        for neigh, count in neigh_counts.items():
            deg_neigh = max(1, len(dict_node_list[neigh]))
            t = min(1 / deg_current, 1 / deg_neigh)
            neighs.append(neigh)
            weights.append(count / total_neigh * t)

        next_node = random.choices(neighs, weights=weights, k=1)[0]

        # Choose next edge incident to next_node, weighted by edge size and previous edge
        next_incident = list(dict_node_list[next_node])
        if not next_incident:
            current_node = next_node
            continue
        edge_weights = []
        for eid in next_incident:
            size_e = max(1, len(dict_edge_list[eid]))
            size_prev = max(1, len(dict_edge_list[current_edge]))
            edge_weights.append(min(1 / size_e, 1 / size_prev))
        next_edge = random.choices(next_incident, weights=edge_weights, k=1)[0]

        sampled_edges.add(next_edge)
        sampled_nodes.update(dict_edge_list[next_edge])

        current_node = next_node
        current_edge = next_edge

    return {
        "sampled_edges": sampled_edges,
        "sampled_nodes": sampled_nodes,
        "params": "JURW",
    }


__all__ = ["expansion_sampling_JURW_MHNorm"]

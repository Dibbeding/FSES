"""
Non-Backtracking Higher-Order Random Walk (NB-HO-RW) baseline.

Adapted from the thesis experiment scripts that re-implemented the method from
“Sampling nodes and hyperedges via random walks on large hypergraphs”.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, Mapping, MutableSet

EdgeId = int
NodeId = int
EdgeList = Mapping[EdgeId, Iterable[NodeId]]
NodeList = Mapping[NodeId, Iterable[EdgeId]]


def expansion_sampling_nbho_rw(
    dict_edge_list: EdgeList,
    dict_node_list: NodeList,
    target_node_count: int,
) -> Dict[str, object]:
    """Run a weighted non-backtracking higher-order random walk until target nodes are covered."""
    sampled_nodes: MutableSet[NodeId] = set()
    sampled_edges: MutableSet[EdgeId] = set()

    node_weight_factor = 10
    edge_weight_factor = 1

    v_current = random.choice(list(dict_node_list.keys()))
    sampled_nodes.add(v_current)

    while len(sampled_nodes) < target_node_count:
        hyperedges = list(dict_node_list[v_current])
        if not hyperedges:
            v_current = random.choice(list(dict_node_list.keys()))
            continue

        edge_weights = [len(dict_edge_list[e]) ** edge_weight_factor for e in hyperedges]
        e_current = random.choices(hyperedges, weights=edge_weights, k=1)[0]
        sampled_edges.add(e_current)

        candidate_nodes = list(dict_edge_list[e_current])
        for n in candidate_nodes:
            sampled_nodes.add(n)

        candidate_nodes = list(set(candidate_nodes) - {v_current})
        if not candidate_nodes:
            v_current = random.choice(list(dict_node_list.keys()))
            continue

        node_weights = [len(dict_node_list[n]) ** node_weight_factor for n in candidate_nodes]
        v_next = random.choices(candidate_nodes, weights=node_weights, k=1)[0]
        sampled_nodes.add(v_next)

        next_hyperedges = list(set(dict_node_list[v_next]) - {e_current})
        if not next_hyperedges:
            v_current = v_next
            continue

        next_edge_weights = [len(dict_edge_list[e]) ** edge_weight_factor for e in next_hyperedges]
        e_next = random.choices(next_hyperedges, weights=next_edge_weights, k=1)[0]
        sampled_edges.add(e_next)

        next_nodes = list(set(dict_edge_list[e_next]) - {v_next})
        if not next_nodes:
            v_current = v_next
            continue

        next_node_weights = [len(dict_node_list[n]) ** node_weight_factor for n in next_nodes]
        v_current = random.choices(next_nodes, weights=next_node_weights, k=1)[0]
        sampled_nodes.add(v_current)

    return {
        "sampled_edges": sampled_edges,
        "sampled_nodes": sampled_nodes,
        "params": f"NBHO_node{node_weight_factor}_edge{edge_weight_factor}",
    }


__all__ = ["expansion_sampling_nbho_rw"]


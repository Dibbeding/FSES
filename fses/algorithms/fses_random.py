"""Baseline random expansion sampler.

Expands a frontier uniformly at random without scoring; used as a control.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, MutableMapping, MutableSet

EdgeId = int
NodeId = int
EdgeList = MutableMapping[EdgeId, MutableSet[NodeId]]
NodeList = MutableMapping[NodeId, MutableSet[EdgeId]]


def expansion_sampling_F_random(
    dict_edge_list: EdgeList,
    dict_node_list: NodeList,
    target_node_count: int,
) -> Dict[str, object]:
    """Sample by randomly expanding the frontier until the target is reached."""

    sampled_nodes: MutableSet[NodeId] = set()
    sampled_edges: MutableSet[EdgeId] = set()

    start_node = random.choice(list(dict_node_list.keys()))
    sampled_nodes.add(start_node)

    frontier = set(dict_node_list[start_node])

    while len(sampled_nodes) < target_node_count and frontier:
        new_edge = random.choice(list(frontier))
        frontier.discard(new_edge)
        sampled_edges.add(new_edge)

        for node in dict_edge_list[new_edge]:
            if node in sampled_nodes:
                continue
            sampled_nodes.add(node)
            for neighbour_edge in dict_node_list[node]:
                frontier.add(neighbour_edge)

    return {
        "sampled_edges": sampled_edges,
        "sampled_nodes": sampled_nodes,
        "params": "random",
    }


__all__ = ["expansion_sampling_F_random"]

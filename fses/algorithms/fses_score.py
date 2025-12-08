"""F-score guided hypergraph expansion sampling (FSES).

Implements the core scoring heuristic from the thesis: expand a frontier of
hyperedges while preferring edges that expose unseen nodes (communities).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, MutableMapping, MutableSet, Optional

# ---------------------------------------------------------------------------#
# Configuration constants
# ---------------------------------------------------------------------------#

UNKNOWN_NODE_SCORE = 1.0
FREQUENCY_DECAY_K = 0.10
SIZE_PENALTY_ALPHA = 0.0
CANDIDATE_LIMIT = 100

# ---------------------------------------------------------------------------#
# Data structures
# ---------------------------------------------------------------------------#

EdgeId = int
NodeId = int


@dataclass
class EdgeScore:
    """Score assigned to a candidate edge and how long it stays fresh."""

    value: float
    refresh_budget: int = 1


Frontier = MutableMapping[EdgeId, EdgeScore]
EdgeList = MutableMapping[EdgeId, MutableSet[NodeId]]
NodeList = MutableMapping[NodeId, MutableSet[EdgeId]]


@dataclass(frozen=True)
class FSESParams:
    """Parameters controlling the F-score expansion heuristic."""

    unknown_node_score: float = UNKNOWN_NODE_SCORE
    k: float = FREQUENCY_DECAY_K
    size_penalty_alpha: float = SIZE_PENALTY_ALPHA
    candidate_limit: int = CANDIDATE_LIMIT
    zero_known: bool = False
    use_refresh: bool = True


# ---------------------------------------------------------------------------#
# Helper functions
# ---------------------------------------------------------------------------#

def decrement_refresh_budget(frontier: Frontier, params: Optional[FSESParams] = None) -> None:
    """Tick down the refresh budget for all edges currently in the frontier."""

    p = params or FSESParams()
    if not p.use_refresh:
        return
    for score in frontier.values():
        score.refresh_budget -= 1


def best_edge(frontier: Frontier) -> EdgeId:
    """Return the edge id with the highest score in the frontier."""

    return max(frontier, key=lambda edge_id: frontier[edge_id].value)


def calc_neighbor_edge_score(
    dict_edge_list: EdgeList,
    dict_node_list: NodeList,
    node_freq: Dict[NodeId, int],
    edge_id: EdgeId,
    params: Optional[FSESParams] = None,
) -> EdgeScore:
    """Score an edge by averaging neighbour-edge scores using node frequencies."""

    neighbour_edges = set()
    for node in dict_edge_list[edge_id]:
        neighbour_edges.update(dict_node_list[node])
    neighbour_edges.discard(edge_id)

    if not neighbour_edges:
        return EdgeScore(0.0, refresh_budget=0)

    scores: List[float] = []
    p = params or FSESParams()
    for neighbour_id in neighbour_edges:
        base = 0.0
        for node in dict_edge_list[neighbour_id]:
            if node not in node_freq:
                base += p.unknown_node_score
            else:
                if p.zero_known:
                    base += 0.0
                else:
                    base += math.exp(-p.k * node_freq[node])

        size = len(dict_edge_list[neighbour_id])
        penalty = 1.0 / (size ** p.size_penalty_alpha) if size > 0 else 1.0
        scores.append(base * penalty)

    mean_score = sum(scores) / len(neighbour_edges)
    return EdgeScore(mean_score, refresh_budget=(1 if p.use_refresh else 10**9))


def enforce_candidate_limit(frontier: Frontier, limit: int) -> None:
    """Keep only the top-``limit`` edges by score in the frontier."""

    if len(frontier) <= limit:
        return

    top_edges = sorted(
        frontier.items(), key=lambda item: item[1].value, reverse=True
    )[:limit]
    frontier.clear()
    frontier.update(top_edges)


def seed_frontier(
    dict_edge_list: EdgeList,
    dict_node_list: NodeList,
    start_node: NodeId,
    node_freq: Dict[NodeId, int],
    params: Optional[FSESParams] = None,
) -> Frontier:
    """Create the initial frontier around the start node."""

    frontier: Frontier = {}

    for edge_id in dict_node_list[start_node]:
        frontier[edge_id] = EdgeScore(0.0, refresh_budget=(1 if (params or FSESParams()).use_refresh else 10**9))
        for node in dict_edge_list[edge_id]:
            node_freq[node] = node_freq.get(node, 0) + 1

    for edge_id in list(frontier.keys()):
        frontier[edge_id] = calc_neighbor_edge_score(
            dict_edge_list,
            dict_node_list,
            node_freq,
            edge_id,
            params,
        )

    enforce_candidate_limit(frontier, (params or FSESParams()).candidate_limit)
    return frontier


def _params_str(p: FSESParams) -> str:
    """Encode parameters into a short token used in filenames."""

    default_k = FREQUENCY_DECAY_K
    default_unseen = UNKNOWN_NODE_SCORE
    default_alpha = SIZE_PENALTY_ALPHA
    default_flim = CANDIDATE_LIMIT

    if p.zero_known:
        return "knownscorezero"

    if abs(p.k - default_k) > 1e-12:
        val = int(round(p.k * 100))  # 0.01->1, 0.10->10, 0.50->50
        return f"knownscoreexp{val:02d}"

    if abs(p.unknown_node_score - default_unseen) > 1e-12:
        val = int(round(p.unknown_node_score))
        return f"unknownscore{val:02d}"

    if abs(p.size_penalty_alpha - default_alpha) > 1e-12:
        mapping = {
            0.10: "hsizea01",
            0.25: "hsizea25",
            0.50: "hsizea05",
            0.90: "hsize90",
        }
        token = mapping.get(round(p.size_penalty_alpha, 2))
        if token:
            return token
        else:
            val = int(round(p.size_penalty_alpha * 100))
            return f"hsizea{val:02d}"

    if p.candidate_limit != default_flim:
        return f"Flim{p.candidate_limit:04d}"

    if not p.use_refresh:
        return "norefresh"

    return "optimal"


# ---------------------------------------------------------------------------#
# Public sampling functions
# ---------------------------------------------------------------------------#

def expansion_sampling_F_score_with_params(
    dict_edge_list: EdgeList,
    dict_node_list: NodeList,
    target_node_count: int,
    params: Optional[FSESParams] = None,
) -> Dict[str, object]:
    """Sample nodes/edges guided by an F-score heuristic using ``params``."""

    sampled_nodes: MutableSet[NodeId] = set()
    sampled_edges: MutableSet[EdgeId] = set()
    node_freq: Dict[NodeId, int] = {}

    start_node = random.choice(list(dict_node_list.keys()))
    sampled_nodes.add(start_node)

    frontier = seed_frontier(dict_edge_list, dict_node_list, start_node, node_freq, params)
    recompute_counter = 0

    while frontier and len(sampled_nodes) < target_node_count:
        edge_id = best_edge(frontier)
        score = frontier[edge_id]

        if (params or FSESParams()).use_refresh and score.refresh_budget < 1:
            frontier[edge_id] = calc_neighbor_edge_score(
                dict_edge_list,
                dict_node_list,
                node_freq,
                edge_id,
                params,
            )
            recompute_counter += 1
            continue

        frontier.pop(edge_id)
        sampled_edges.add(edge_id)

        decrement_refresh_budget(frontier, params)

        candidate_edges: MutableSet[EdgeId] = set()
        for node in dict_edge_list[edge_id]:
            sampled_nodes.add(node)
            for neighbour_edge in dict_node_list[node]:
                candidate_edges.add(neighbour_edge)
                for neighbour_node in dict_edge_list[neighbour_edge]:
                    node_freq[neighbour_node] = node_freq.get(neighbour_node, 0) + 1

        for candidate in candidate_edges:
            if candidate in sampled_edges or candidate in frontier:
                continue
            frontier[candidate] = calc_neighbor_edge_score(
                dict_edge_list,
                dict_node_list,
                node_freq,
                candidate,
                params,
            )

        enforce_candidate_limit(frontier, (params or FSESParams()).candidate_limit)

    print("Number of score recomputations:", recompute_counter)
    return {
        "sampled_edges": sampled_edges,
        "sampled_nodes": sampled_nodes,
        "params": _params_str(params or FSESParams()),
    }


def expansion_sampling_F_score(
    dict_edge_list: EdgeList,
    dict_node_list: NodeList,
    target_node_count: int,
) -> Dict[str, object]:
    """Default F-score sampling using module-level defaults."""

    return expansion_sampling_F_score_with_params(
        dict_edge_list,
        dict_node_list,
        target_node_count,
        FSESParams(),
    )


__all__ = [
    "expansion_sampling_F_score",
    "expansion_sampling_F_score_with_params",
    "FSESParams",
    "EdgeScore",
    "best_edge",
    "decrement_refresh_budget",
    "enforce_candidate_limit",
]

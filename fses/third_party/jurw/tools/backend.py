from __future__ import annotations

import random
from tools.Hypergraph import Hypergraph

HGraph = type(Hypergraph())


class HypernetxBackEnd:
    """Minimal backend used by the JURW sampler."""

    def get_number_of_nodes(self, graph: HGraph) -> int:
        return graph.number_of_nodes()

    def get_number_of_edges(self, graph: HGraph) -> int:
        return graph.number_of_edges()

    def get_nodes(self, graph: HGraph):
        return graph.get_nodes()

    def get_edges(self, graph: HGraph):
        return graph.get_edges()

    def get_node(self, graph: HGraph, edge: int):
        return graph.get_node(edge)

    def get_edge(self, graph: HGraph, node: int):
        return graph.get_edge(node)

    def get_degree(self, graph: HGraph, node: int) -> int:
        return graph.get_degree(node)

    def get_edges_length(self, graph: HGraph, edge: int) -> int:
        return len(graph.get_node(edge))

    def get_random_edge(self, graph: HGraph, node: int) -> int:
        return random.choice(graph.get_edge(node))

    def get_random_node(self, graph: HGraph, edge: int) -> int:
        return random.choice(graph.get_node(edge))

    def get_hypergraph(self, graph: HGraph):
        return graph.get_hypergraph()

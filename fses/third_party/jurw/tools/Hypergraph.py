from __future__ import annotations

import copy


from typing import Optional


class Hypergraph:
    """
    Lightweight hypergraph container used by the JURW reference code.

    The file format expected is one hyperedge per line with space-separated 0-based node ids.
    """

    def __init__(self, filename: Optional[str] = None):
        self.hyperEdge: list[list[int]] = []
        self.hyperNode: list[list[int]] = []
        if filename is not None:
            edges: list[list[int]] = []
            with open(filename, encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    edges.append([int(tok) for tok in line.split()])
            self.hyperEdge = edges

            node_to_edges: dict[int, list[int]] = {}
            for edge_id, edge in enumerate(edges):
                for node in edge:
                    node_to_edges.setdefault(node, []).append(edge_id)

            max_node = max(node_to_edges.keys()) if node_to_edges else -1
            dense_nodes = []
            for idx in range(max_node + 1):
                dense_nodes.append(node_to_edges.get(idx, []))
            self.hyperNode = copy.deepcopy(dense_nodes)

    def number_of_nodes(self) -> int:
        return len(self.hyperNode)

    def number_of_edges(self) -> int:
        return len(self.hyperEdge)

    def get_nodes(self) -> list[int]:
        return [idx for idx in range(len(self.hyperNode))]

    def get_edges(self) -> list[int]:
        return [idx for idx in range(len(self.hyperEdge))]

    def get_degree(self, node: int) -> int:
        return len(self.hyperNode[node])

    def get_length_edge(self, edge: int) -> int:
        return len(self.hyperEdge[edge])

    def get_node(self, edge: int) -> list[int]:
        return self.hyperEdge[edge]

    def get_edge(self, node: int) -> list[int]:
        return self.hyperNode[node]

    def get_hypergraph(self) -> tuple[list[list[int]], list[list[int]]]:
        return self.hyperEdge, self.hyperNode

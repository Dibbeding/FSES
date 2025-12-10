from __future__ import annotations

import itertools
import random
from typing import Tuple, Union

import numpy as np
from tools import Hypergraph, SampleData, Sampler

HGraph = type(Hypergraph())


class RandomWalkSamplerWithMHNormalization(Sampler):
    """
    Reference implementation of the Joint Unbiased Random Walk (Metropolis-Hastings normalized).
    Slightly trimmed to remove unused code but algorithmically identical to the original.
    """

    def __init__(self, number_of_nodes: int = 100, seed: int = 22, step: int = 100):
        super().__init__()
        self._sample_data = SampleData(step)
        self._number_of_nodes = number_of_nodes
        self._seed = seed
        self._set_seed()
        self._sample_count = 1
        self._N: dict[int, int] = {}
        self._NeiCount: dict[int, dict[int, int]] = {}

    def _create_initial_node_set(self, graph, start_node):
        if start_node is not None:
            if 0 <= start_node < self.backend.get_number_of_nodes(graph):
                self._current_node = start_node
                self._current_edge = self.backend.get_random_edge(graph, self._current_node)
                self._sample_data.sample_node(self._current_node)
                self._sample_data.sample_edge(self._current_edge)
                self._sample_data.sample_unique_node(self._current_node)
                self._sample_data.sample_unique_edge(self._current_edge)
            else:
                raise ValueError("Starting node index is out of range.")
        else:
            self._current_node = random.choice(range(self.backend.get_number_of_nodes(graph)))
            self._current_edge = self.backend.get_random_edge(graph, self._current_node)
            self._sample_data.sample_node(self._current_node)
            self._sample_data.sample_edge(self._current_edge)
            self._sample_data.sample_unique_node(self._current_node)
            self._sample_data.sample_unique_edge(self._current_edge)

    def _do_a_step(self, graph):
        if self._current_node not in self._N:
            self._NeiCount[self._current_node] = {}
            for edge_id in self.backend.get_edge(graph, self._current_node):
                edge = self.backend.get_node(graph, edge_id)
                for neighbor in edge:
                    if neighbor != self._current_node:
                        self._NeiCount[self._current_node][neighbor] = (
                            self._NeiCount[self._current_node].get(neighbor, 0) + 1
                        )
            self._N[self._current_node] = sum(self._NeiCount[self._current_node].values())

        vertices = []
        weights = []
        for neighbor, count in self._NeiCount[self._current_node].items():
            vertices.append(neighbor)
            t = min(
                1 / len(self.backend.get_edge(graph, self._current_node)),
                1 / len(self.backend.get_edge(graph, neighbor)),
            )
            weights.append(count / self._N[self._current_node] * t)

        self._current_node = random.choices(vertices, weights=weights, k=1)[0]
        self._sample_data.access_unique_node(self._current_node)
        self._sample_data.sample_node(self._current_node)
        self._sample_data.sample_unique_node(self._current_node)

        edges = []
        edge_weights = []
        for edge_id in self.backend.get_edge(graph, self._current_node):
            edges.append(edge_id)
            t = min(
                1 / len(self.backend.get_node(graph, edge_id)),
                1 / len(self.backend.get_node(graph, self._current_edge)),
            )
            edge_weights.append(t)

        self._current_edge = random.choices(edges, weights=edge_weights, k=1)[0]
        self._sample_data.sample_edge(self._current_edge)
        self._sample_data.sample_unique_edge(self._current_edge)

    def sample(self, graph: HGraph, start_node: int = None) -> Tuple[list, list, list, list, list]:
        self._deploy_backend(graph)
        self._create_initial_node_set(graph, start_node)
        while self._sample_count < self._number_of_nodes:
            self._do_a_step(graph)
            self._sample_count += 1
            self._sample_data.record(self._sample_count)
        self._sample_data.record_end(self._sample_count)
        return self._sample_data

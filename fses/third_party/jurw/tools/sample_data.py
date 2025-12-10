from __future__ import annotations

import numpy as np


class SampleData:
    """Tracks sampled nodes/edges for the JURW reference sampler."""

    def __init__(self, step: int = 100):
        self._step = step
        self._sample_node = []
        self._sample_edge = []
        self._sample_unique_node = []
        self._sample_unique_edge = []
        self._sample_data = []

    def sample_node(self, node: int):
        self._sample_node.append(node)

    def sample_edge(self, edge: int):
        self._sample_edge.append(edge)

    def sample_unique_node(self, node: int):
        self._sample_unique_node.append(node)

    def sample_unique_edge(self, edge: int):
        self._sample_unique_edge.append(edge)

    def access_unique_node(self, node: int):
        self._sample_unique_node.append(node)

    def record(self, index: int):
        if index % self._step == 0:
            self._sample_data.append(self._record_template(index))

    def record_end(self, index: int):
        self._sample_data.append(self._record_template(index))

    def _record_template(self, index: int):
        return {
            "Step": index,
            "SampleNodes": self._sample_node.copy(),
            "SampleEdges": self._sample_edge.copy(),
            "SampleUniqueNode": np.unique(self._sample_unique_node).tolist(),
            "SampleUniqueEdge": np.unique(self._sample_unique_edge).tolist(),
        }

    def get_sample_edge(self):
        return self._sample_unique_edge

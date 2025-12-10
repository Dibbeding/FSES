from __future__ import annotations

import random
import numpy as np
from tools.backend import HypernetxBackEnd

HGraph = object


class Sampler:
    """Base sampler from the JURW reference implementation."""

    def __init__(self):
        self.backend: HypernetxBackEnd | None = None

    def sample(self):
        raise NotImplementedError

    def _set_seed(self) -> None:
        random.seed(self._seed)
        np.random.seed(self._seed)

    def _deploy_backend(self, graph: HGraph) -> None:
        self.backend = HypernetxBackEnd()

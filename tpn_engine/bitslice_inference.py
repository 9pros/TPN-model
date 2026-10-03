"""
Bitsliced inference integration.

The core idea: when N independent inputs share the same weight matrix
(as happens for batching, or for the multiple trajectories/intents the
TPN engine evaluates), the weights should be loaded and traversed ONCE
and applied to all N inputs, rather than reloaded N times.

This is weight-stationary batching — the same principle that makes
SIMD and tensor-core batching efficient. `weight_traversals` counts how
many times the weight matrix is read, which is the quantity that maps to
real memory-bandwidth savings on hardware.
"""

import math
from typing import List, Optional
from .phase import PhaseWeight


class BitslicedLinear:
    """
    A linear layer that applies one weight matrix to N inputs in a single
    traversal.

    Args:
        rows: Output dimension.
        cols: Input dimension.
        num_slices: Number of independent inputs processed per traversal.
    """

    def __init__(self, rows: int, cols: int, num_slices: int = 4,
                 seed: Optional[int] = None):
        self.rows = rows
        self.cols = cols
        self.num_slices = num_slices

        import random
        if seed is not None:
            random.seed(seed)
        self.weights: List[List[PhaseWeight]] = [
            [PhaseWeight(math.pi / 2 + random.gauss(0, 0.1)) for _ in range(cols)]
            for _ in range(rows)
        ]
        self.weight_traversals = 0
        self.reset_counters()

    def reset_counters(self) -> None:
        self.weight_traversals = 0

    def reference_forward(self, vector: List[float]) -> List[float]:
        """Single-input reference matvec (no bitslicing)."""
        result = [0.0] * self.rows
        for i in range(self.rows):
            s = 0.0
            row = self.weights[i]
            for j in range(self.cols):
                s += row[j].value * vector[j]
            result[i] = s
        return result

    def forward(self, inputs: List[List[float]]) -> List[List[float]]:
        """
        Apply the weight matrix to a batch of inputs.

        The outer loop is over weights (loaded once), the inner loop is
        over the batch — this is the bitsliced/weight-stationary ordering.
        """
        batch = len(inputs)
        # Pre-transpose inputs so the inner loop over batch is contiguous
        cols = self.cols
        # results[j][i] for input j, output i
        results = [[0.0] * self.rows for _ in range(batch)]

        # Single traversal of the weight matrix
        self.weight_traversals += 1
        for i in range(self.rows):
            row = self.weights[i]
            for j in range(cols):
                w = row[j].value
                for b in range(batch):
                    results[b][i] += w * inputs[b][j]
        return results


class BitslicedAttention:
    """
    Attention that evaluates N (q, k, v) triples in one pass.

    Used by the TPN engine to evaluate multiple trajectories/intents in
    parallel, sharing the scoring work.
    """

    def __init__(self, head_dim: int, num_slices: int = 4):
        self.head_dim = head_dim
        self.num_slices = num_slices
        self.weight_traversals = 0

    def reset_counters(self) -> None:
        self.weight_traversals = 0

    @staticmethod
    def _score(q: List[float], k: List[float]) -> float:
        d = len(q)
        return sum(q[i] * k[i] for i in range(d)) / math.sqrt(d)

    def reference_forward(self, q: List[float], k: List[float],
                          v: List[float]) -> List[float]:
        """Single-triple reference attention."""
        s = self._score(q, k)
        fitness = math.cos(s - math.pi / 4.0) ** 2
        return [fitness * vi for vi in v]

    def forward(self, q: List[float], k: List[float], v: List[float]) -> List[float]:
        """Single-triple forward (delegates to reference math)."""
        return self.reference_forward(q, k, v)

    def forward_batch(self, qs: List[List[float]], ks: List[List[float]],
                      vs: List[List[float]]) -> List[List[float]]:
        """
        Evaluate a batch of (q, k, v) triples.

        The score computation is organized so the shared dimension is
        traversed once across the whole batch.
        """
        batch = len(qs)
        d = self.head_dim
        sqrt_d = math.sqrt(d)

        # Compute all scores in a single traversal over the shared dim
        self.weight_traversals += 1
        scores = [0.0] * batch
        for i in range(d):
            for b in range(batch):
                scores[b] += qs[b][i] * ks[b][i]
        for b in range(batch):
            scores[b] /= sqrt_d

        # Apply fitness and scale values
        out = []
        for b in range(batch):
            fitness = math.cos(scores[b] - math.pi / 4.0) ** 2
            out.append([fitness * vs[b][i] for i in range(d)])
        return out

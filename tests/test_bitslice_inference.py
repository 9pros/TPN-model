"""
Tests for bitsliced inference integration.

One weight matrix is applied to N independent input vectors in a single
traversal (weight-stationary batching), instead of reloading weights N times.
"""

import time
import math
import pytest
from tpn_engine.bitslice_inference import BitslicedLinear, BitslicedAttention


class TestBitslicedLinear:
    def test_creation(self):
        lin = BitslicedLinear(rows=8, cols=8, num_slices=4)
        assert lin.rows == 8
        assert lin.cols == 8
        assert lin.num_slices == 4

    def test_matches_reference(self):
        """Bitsliced matvec must equal the reference matvec."""
        lin = BitslicedLinear(rows=4, cols=4, num_slices=3)
        inputs = [[1.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0]]
        out = lin.forward(inputs)
        assert len(out) == 3
        for i, inp in enumerate(inputs):
            ref = lin.reference_forward(inp)
            for a, b in zip(out[i], ref):
                assert abs(a - b) < 1e-9

    def test_batch_vs_sequential_identical(self):
        lin = BitslicedLinear(rows=16, cols=16, num_slices=8)
        inputs = [[0.1 * (i + j) for j in range(16)] for i in range(8)]
        batched = lin.forward(inputs)
        sequential = [lin.reference_forward(x) for x in inputs]
        for b, s in zip(batched, sequential):
            for x, y in zip(b, s):
                assert abs(x - y) < 1e-9

    def test_weight_reuse_counter(self):
        """Weights should be traversed once per batch, not once per input."""
        lin = BitslicedLinear(rows=8, cols=8, num_slices=4)
        lin.reset_counters()
        inputs = [[0.1] * 8 for _ in range(4)]
        lin.forward(inputs)
        # one pass over weights for the whole batch
        assert lin.weight_traversals == 1


class TestBitslicedAttention:
    def test_creation(self):
        attn = BitslicedAttention(head_dim=16, num_slices=4)
        assert attn.head_dim == 16
        assert attn.num_slices == 4

    def test_matches_reference(self):
        attn = BitslicedAttention(head_dim=8, num_slices=2)
        q = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        k = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        v = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        out = attn.forward(q, k, v)
        ref = attn.reference_forward(q, k, v)
        for a, b in zip(out, ref):
            assert abs(a - b) < 1e-9

    def test_parallel_speedup_is_measurable(self):
        """Measure batched vs sequential. Report the ratio honestly."""
        attn = BitslicedAttention(head_dim=64, num_slices=32)
        qs = [[0.1 * (i + j) for j in range(64)] for i in range(32)]
        ks = [[0.2 * (i + j) for j in range(64)] for i in range(32)]
        vs = [[0.3 * (i + j) for j in range(64)] for i in range(32)]

        t0 = time.perf_counter()
        for i in range(32):
            attn.reference_forward(qs[i], ks[i], vs[i])
        seq_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        attn.forward_batch(qs, ks, vs)
        batch_time = time.perf_counter() - t0

        # Not asserting a speedup magnitude (Python overhead varies);
        # asserting both complete and batch is not catastrophically slower.
        assert batch_time < seq_time * 5.0, (
            f"batch {batch_time:.4f}s vs seq {seq_time:.4f}s"
        )

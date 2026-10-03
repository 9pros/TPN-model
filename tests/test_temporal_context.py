"""
Tests for temporal (time-based) context.

The claim: context is encoded as a fixed-size phase superposition, so
memory and per-token cost are O(dim) -- independent of context length --
instead of the O(n*d) KV cache and O(n^2*d) attention of a transformer.
"""

import math
import cmath
import random
import pytest
from tpn_engine.temporal_context import TemporalContext, TemporalMemory


class TestTemporalContext:
    def test_creation(self):
        ctx = TemporalContext(dim=64, num_frequencies=16)
        assert ctx.dim == 64
        assert ctx.length == 0

    def test_update_increments_length(self):
        ctx = TemporalContext(dim=64, num_frequencies=16)
        ctx.update([0.1] * 64, arrival_time=0.0)
        assert ctx.length == 1
        ctx.update([0.2] * 64, arrival_time=1.0)
        assert ctx.length == 2

    def test_memory_is_fixed_size(self):
        """Memory footprint must NOT grow with context length."""
        ctx = TemporalContext(dim=64, num_frequencies=16)
        size_after_1 = ctx.memory_bytes()
        for i in range(1000):
            ctx.update([0.01 * (i % 7)] * 64, arrival_time=float(i))
        assert ctx.memory_bytes() == size_after_1

    def test_state_is_bounded(self):
        """State magnitude must stay bounded as context grows."""
        ctx = TemporalContext(dim=64, num_frequencies=16, forgetting=0.999)
        for i in range(5000):
            ctx.update([0.1] * 64, arrival_time=float(i))
        assert ctx.state_norm() < 1e6

    def test_retrieve_returns_vector(self):
        ctx = TemporalContext(dim=64, num_frequencies=16)
        for i in range(10):
            ctx.update([0.1] * 64, arrival_time=float(i))
        out = ctx.retrieve(query_time=10.0)
        assert len(out) == 64

    def test_exact_recall_at_matching_time(self):
        """A single written vector must be recovered almost exactly."""
        ctx = TemporalContext(dim=32, num_frequencies=32, forgetting=1.0,
                              window=1e6)
        vec = [random.gauss(0, 1) for _ in range(32)]
        ctx.update(vec, arrival_time=5.0)
        out = ctx.retrieve(query_time=5.0)
        num = sum(a * b for a, b in zip(vec, out))
        den = (math.sqrt(sum(a * a for a in vec)) *
               math.sqrt(sum(b * b for b in out))) or 1.0
        assert num / den > 0.99

    def test_wrong_time_gives_lower_recall(self):
        """Querying at the wrong time must retrieve less of the vector."""
        ctx = TemporalContext(dim=32, num_frequencies=32, forgetting=1.0,
                              window=1e6)
        vec = [random.gauss(0, 1) for _ in range(32)]
        ctx.update(vec, arrival_time=5.0)

        def cos(a, b):
            n = sum(x * y for x, y in zip(a, b))
            d = (math.sqrt(sum(x * x for x in a)) *
                 math.sqrt(sum(y * y for y in b))) or 1.0
            return n / d

        right = cos(vec, ctx.retrieve(query_time=5.0))
        wrong = cos(vec, ctx.retrieve(query_time=250.0))
        assert right > wrong

    def test_relative_position_is_phase(self):
        """Identical tokens at different times must carry different phase."""
        ctx = TemporalContext(dim=8, num_frequencies=8)
        ctx.update([1.0] * 8, arrival_time=0.0)
        s0 = ctx.phase_signature()
        ctx2 = TemporalContext(dim=8, num_frequencies=8)
        ctx2.update([1.0] * 8, arrival_time=1.0)
        s1 = ctx2.phase_signature()
        diff = abs(cmath.phase(s0[0][0]) - cmath.phase(s1[0][0]))
        assert diff > 1e-9

    def test_compression_ratio_grows_with_length(self):
        """Memory is constant, so compression vs KV cache grows with n."""
        ratios = []
        for n in [256, 4096, 65536]:
            ctx = TemporalContext(dim=64, num_frequencies=64)
            for i in range(n):
                ctx.update([0.1] * 64, arrival_time=float(i))
            ratios.append(ctx.compression_ratio())
        # ratio = n / (2*dim); must be strictly increasing with n
        assert ratios[0] < ratios[1] < ratios[2]
        assert ratios[2] > 10.0


class TestTemporalMemory:
    def test_read_returns_score(self):
        mem = TemporalMemory(dim=32, num_frequencies=32, window=1e6,
                             forgetting=1.0)
        vec = [0.0] * 32
        vec[0] = 1.0
        mem.write(vec, arrival_time=0.0)
        _, score = mem.read(vec, query_time=0.0)
        assert score > 0.9

    def test_recent_beats_old_with_forgetting(self):
        """With forgetting, a recent item must be recalled better than an old one."""
        mem = TemporalMemory(dim=32, num_frequencies=32, window=1e6,
                             forgetting=0.99)
        old = [random.gauss(0, 1) for _ in range(32)]
        mem.write(old, arrival_time=0.0)
        new = [random.gauss(0, 1) for _ in range(32)]
        mem.write(new, arrival_time=500.0)

        _, new_score = mem.read(new, query_time=500.0)
        _, old_score = mem.read(old, query_time=0.0)
        assert new_score > old_score

    def test_capacity_grows_with_dim(self):
        """A larger dim must hold more items at usable recall."""
        def max_items(d):
            best = 0
            for n in [8, 32, 128]:
                s = 0.0
                for _ in range(8):
                    mem = TemporalMemory(dim=d, num_frequencies=min(32, d),
                                         window=1e6, forgetting=1.0)
                    vecs = []
                    for i in range(n):
                        v = [random.gauss(0, 1) for _ in range(d)]
                        vecs.append(v)
                        mem.write(v, arrival_time=float(i))
                    _, sc = mem.read(vecs[-1], query_time=float(n - 1))
                    s += sc
                if s / 8 > 0.3:
                    best = n
            return best
        assert max_items(256) >= max_items(32)

    def test_graceful_degradation_no_collapse(self):
        """Scores must stay finite and in range as context grows."""
        for n in [4, 64, 256, 1024]:
            mem = TemporalMemory(dim=32, num_frequencies=32,
                                 window=1e6, forgetting=0.9999)
            for i in range(n):
                mem.write([random.gauss(0, 1) for _ in range(32)],
                          arrival_time=float(i))
            _, score = mem.read([random.gauss(0, 1) for _ in range(32)],
                                query_time=float(n - 1))
            assert -1.0 <= score <= 1.0

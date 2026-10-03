"""
Temporal (time-based) context for TPN.

Context is encoded as a fixed-size *matrix* state (linear attention /
associative memory), not a growing KV cache:

    state[i,k] = sum_j  decay(t_j) * key_j[i] * v_j[k]

where the key is a phase-coded time signature:

    key_j[i] = exp(i * omega_i * t_j) / sqrt(dim)

Retrieval unbinds the state with the query-time key:

    out[k] = Re( sum_i conj(key_t[i]) * state[i,k] )
           = sum_j decay(t_j) * a(t_j, t) * v_j[k]

    a(t_j, t) = (1/dim) * sum_i exp(i * omega_i * (t_j - t))

For the token whose arrival equals t, a = 1 and its value is recovered
exactly. For other tokens a is the (small) autocorrelation of the
frequency bank, so they contribute bounded cross-talk.

Properties:
- memory is O(dim^2), CONSTANT in context length (no KV cache growth)
- per-token update is O(dim^2)
- relative position is encoded as a phase difference
- capacity scales with dim: a d-dimensional state holds ~O(dim) items,
  because the state is an outer-product memory, not a vector super-
  position. (A vector superposition has O(1) capacity -- verified.)

This is the same capacity class as linear attention and modern SSMs. It
is finite capacity, not perfect infinite recall.
"""

import math
import cmath
import random as _random_module
from typing import List, Optional, Tuple


def _random_frequency(rng, window: float) -> float:
    """
    Draw one angular frequency (rad / time unit) in a resolvable band.

    Upper bound ~ pi (Nyquist); lower bound set by the window we want to
    resolve (a full cycle per `window` time units). Random within the band
    keeps keys near-orthogonal across dimensions.
    """
    f_hi = 0.5                     # cycles / time unit (Nyquist)
    f_lo = 0.5 / max(2.0, window)  # cycles / time unit (window resolution)
    f = rng.uniform(f_lo, f_hi)
    return 2.0 * math.pi * f


class TemporalContext:
    """
    Fixed-size temporal context with an outer-product (matrix) state.

    Args:
        dim: Vector dimension; capacity scales with this.
        num_frequencies: Distinct frequency slots (<= dim).
        window: Number of time steps the frequency bank resolves.
        forgetting: Exponential decay per unit time (1.0 = no forgetting).
    """

    def __init__(self, dim: int = 64, num_frequencies: int = 64,
                 window: float = 128.0, forgetting: float = 0.999,
                 seed: int = 0):
        self.dim = dim
        self.num_frequencies = min(num_frequencies, dim)
        self.window = window
        self.forgetting = forgetting

        # Frequency bank. Random per-dimension frequencies in (0, pi] give
        # near-orthogonal time keys (cross-talk ~ 1/sqrt(dim)), which is
        # what makes capacity scale with dim. A log-spaced bank is
        # structured and has a slowly-decaying autocorrelation, which
        # collapses capacity to O(1) -- avoid it here.
        import random as _random
        rng = _random.Random(seed)
        self.omegas = [_random_frequency(rng, window) for _ in range(dim)]

        # Fixed-size matrix state (dim x dim), complex.
        self._state: List[List[complex]] = [[0j] * dim for _ in range(dim)]
        self._mass: float = 0.0
        self._length: int = 0
        self._last_time: float = 0.0

    # ------------------------------------------------------------------ #

    @property
    def length(self) -> int:
        return self._length

    def _key(self, t: float) -> List[complex]:
        """Phase-coded time signature (unit-ish norm)."""
        s = 1.0 / math.sqrt(self.dim)
        return [cmath.exp(1j * self.omegas[i] * t) * s for i in range(self.dim)]

    def update(self, vector: List[float], arrival_time: float) -> None:
        """Write a vector at a given arrival time. O(dim^2), O(1) memory."""
        if len(vector) != self.dim:
            raise ValueError(f"vector dim {len(vector)} != {self.dim}")

        dt = max(0.0, arrival_time - self._last_time)
        decay = self.forgetting ** dt
        if decay != 1.0:
            for i in range(self.dim):
                row = self._state[i]
                for k in range(self.dim):
                    row[k] *= decay
            self._mass *= decay

        key = self._key(arrival_time)
        for i in range(self.dim):
            ki = key[i]
            row = self._state[i]
            for k in range(self.dim):
                row[k] += ki * vector[k]

        self._mass += 1.0
        self._length += 1
        self._last_time = arrival_time

    def retrieve(self, query_time: float) -> List[float]:
        """Unbind the state with the query-time key. Returns a real vector."""
        key = self._key(query_time)
        out = [0.0] * self.dim
        for i in range(self.dim):
            ck = key[i].conjugate()
            row = self._state[i]
            for k in range(self.dim):
                out[k] += (ck * row[k]).real
        return out

    def phase_signature(self) -> List[List[complex]]:
        return [list(row) for row in self._state]

    def state_norm(self) -> float:
        return math.sqrt(sum(abs(z) ** 2 for row in self._state for z in row))

    def memory_bytes(self) -> int:
        """Fixed footprint: dim^2 complex128 + scalars. Constant in n."""
        return self.dim * self.dim * 16 + 64

    def compression_ratio(self) -> float:
        """Ratio of a full KV cache (K+V, fp32) to this footprint."""
        if self.memory_bytes() == 0:
            return 0.0
        kv_bytes = self._length * self.dim * 2 * 4
        return kv_bytes / self.memory_bytes()

    def effective_context_window(self) -> float:
        if self.forgetting >= 1.0:
            return float("inf")
        return -1.0 / math.log(self.forgetting)


class TemporalMemory:
    """Temporal memory with a clean read/score API for retrieval quality."""

    def __init__(self, dim: int = 32, num_frequencies: int = 32,
                 window: float = 128.0, forgetting: float = 0.999):
        self.ctx = TemporalContext(dim=dim, num_frequencies=num_frequencies,
                                   window=window, forgetting=forgetting)
        self.dim = dim

    def write(self, vector: List[float], arrival_time: float) -> None:
        self.ctx.update(vector, arrival_time)

    def read(self, query_vector: List[float], query_time: float) -> Tuple[List[float], float]:
        """Read by time; score = cosine similarity with the query vector."""
        retrieved = self.ctx.retrieve(query_time)
        num = sum(a * b for a, b in zip(query_vector, retrieved))
        qn = math.sqrt(sum(a * a for a in query_vector)) or 1.0
        rn = math.sqrt(sum(b * b for b in retrieved)) or 1.0
        return retrieved, num / (qn * rn)

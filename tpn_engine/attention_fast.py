"""
Vectorized TPN attention (numpy fast path).

This module provides the same math as ``tpn_engine.attention`` but with
numpy arrays instead of nested Python lists, so the score matmul, the
fitness softmax and the value mix are each a single C-level call.

The reference implementation in ``attention.py`` remains the source of
truth for the math; ``tests/test_attention_fast.py`` asserts the two
agree elementwise.
"""

import math

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional
    np = None


def _require_numpy():
    if np is None:
        raise ImportError(
            "numpy is required for the vectorized attention path. "
            "Install it with: pip install numpy"
        )


def tpn_softmax_fast(scores):
    """
    Fitness-based softmax over the last axis, vectorized.

    For each row:

        theta_j = (π/2) * (s_j - min) / (max - min)
        w_j     = cos²(theta_j - π/4)
        out_j   = w_j / sum(w)

    Degenerate rows (max == min) use theta = π/4, which gives
    fitness = cos²(0) = 1 for every element -> uniform weights,
    matching the reference implementation.
    """
    _require_numpy()
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim == 1:
        s = s[None, :]

    lo = s.min(axis=-1, keepdims=True)
    hi = s.max(axis=-1, keepdims=True)
    span = hi - lo

    # Avoid divide-by-zero; degenerate rows are overwritten below.
    safe_span = np.where(span > 0.0, span, 1.0)
    theta = (math.pi / 2.0) * (s - lo) / safe_span
    theta = np.where(span > 0.0, theta, math.pi / 4.0)

    fit = np.cos(theta - math.pi / 4.0) ** 2
    total = fit.sum(axis=-1, keepdims=True)
    # If every fitness is zero (cannot happen for cos², but be safe),
    # fall back to uniform.
    total = np.where(total > 0.0, total, 1.0)
    return fit / total


def standard_softmax_fast(scores):
    """Numerically stable exp softmax over the last axis, vectorized."""
    _require_numpy()
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim == 1:
        s = s[None, :]
    shifted = s - s.max(axis=-1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=-1, keepdims=True)


def compute_scores_fast(q, k):
    """Attention scores Q @ K^T / sqrt(d), vectorized."""
    _require_numpy()
    qa = np.asarray(q, dtype=np.float64)
    ka = np.asarray(k, dtype=np.float64)
    d = qa.shape[-1]
    return (qa @ ka.T) / math.sqrt(d)


def compute_output_fast(weights, v):
    """Attention weights @ V, vectorized."""
    _require_numpy()
    wa = np.asarray(weights, dtype=np.float64)
    va = np.asarray(v, dtype=np.float64)
    return wa @ va


def attention_fast(q, k, v):
    """Full TPN attention forward pass, vectorized."""
    scores = compute_scores_fast(q, k)
    weights = tpn_softmax_fast(scores)
    return compute_output_fast(weights, v)


def attention_standard_fast(q, k, v):
    """Standard-attention forward pass, vectorized (for comparison)."""
    scores = compute_scores_fast(q, k)
    weights = standard_softmax_fast(scores)
    return compute_output_fast(weights, v)


class TPNAttentionFast:
    """
    Vectorized TPN attention.

    Same public surface as ``TPNAttention`` (compute_scores,
    compute_output, forward, forward_standard) but operating on numpy
    arrays. Accepts nested lists as well; they are converted on entry.
    """

    @staticmethod
    def compute_scores(q, k):
        return compute_scores_fast(q, k)

    @staticmethod
    def compute_output(weights, v):
        return compute_output_fast(weights, v)

    @staticmethod
    def forward(q, k, v):
        return attention_fast(q, k, v)

    @staticmethod
    def forward_standard(q, k, v):
        return attention_standard_fast(q, k, v)

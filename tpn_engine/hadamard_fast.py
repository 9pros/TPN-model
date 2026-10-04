"""
Vectorized Hadamard transform (numpy fast path).

The reference implementation in ``hadamard.py`` materializes the full
N x N Sylvester-Walsh matrix as nested Python lists and does an O(N^2)
matvec. That is both slow and memory-hungry: at block_size=1024 the
matrix alone is over a million Python floats.

This module computes the *same* transform in O(N log N) using the
butterfly (fast Walsh-Hadamard) algorithm, vectorized across all
butterfly groups with numpy. The result is identical to the matrix
form up to floating-point rounding; ``tests/test_hadamard_fast.py``
asserts the two agree elementwise.
"""

import math

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional
    np = None


def _require_numpy():
    if np is None:
        raise ImportError(
            "numpy is required for the vectorized Hadamard path. "
            "Install it with: pip install numpy"
        )


def is_power_of_two(n: int) -> bool:
    """True when n is a positive power of two."""
    return n > 0 and (n & (n - 1)) == 0


def next_power_of_two(n: int) -> int:
    """Smallest power of two >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def fwht(x):
    """
    Unnormalized fast Walsh-Hadamard transform (in place on a copy).

    The unnormalized transform is its own inverse up to a factor of n:
    applying it twice returns n * x. Normalization is applied by the
    caller.

    Args:
        x: 1-D array of length n, where n is a power of two.

    Returns:
        Transformed array (new array; input is not modified).
    """
    _require_numpy()
    x = np.asarray(x, dtype=np.float64).copy()
    n = x.shape[0]
    if not is_power_of_two(n):
        raise ValueError(f"length must be a power of two, got {n}")

    h = 1
    while h < n:
        # View as (groups, 2, h) and apply one butterfly stage to every
        # group simultaneously.
        x = x.reshape(-1, 2, h)
        a = x[:, 0, :].copy()
        b = x[:, 1, :].copy()
        x[:, 0, :] = a + b
        x[:, 1, :] = a - b
        x = x.reshape(-1)
        h *= 2
    return x


def hadamard_transform(vector, block_size=None):
    """
    Normalized Hadamard transform, matching HadamardTransform.transform.

    The input is zero-padded or truncated to ``block_size``, then
    transformed and scaled by 1/sqrt(block_size).

    Args:
        vector: input sequence of floats.
        block_size: transform size (power of two). Defaults to
            len(vector) rounded up to a power of two.

    Returns:
        numpy array of length block_size.
    """
    _require_numpy()
    v = np.asarray(vector, dtype=np.float64).ravel()

    if block_size is None:
        block_size = next_power_of_two(max(1, v.shape[0]))

    if not is_power_of_two(block_size):
        raise ValueError(
            f"block_size must be a power of two, got {block_size}"
        )

    n = v.shape[0]
    if n < block_size:
        v = np.concatenate([v, np.zeros(block_size - n, dtype=np.float64)])
    elif n > block_size:
        v = v[:block_size]

    return fwht(v) / math.sqrt(block_size)


def hadamard_inverse(vector, block_size=None):
    """
    Inverse normalized Hadamard transform.

    The normalized transform is its own inverse, so this is the same
    operation as the forward transform.
    """
    return hadamard_transform(vector, block_size=block_size)


class FastHadamardTransform:
    """
    Drop-in replacement for ``HadamardTransform`` using the O(N log N)
    butterfly algorithm.

    Exposes the same ``block_size``, ``transform`` and
    ``inverse_transform`` surface, so it can be swapped in anywhere the
    reference class is used.
    """

    def __init__(self, block_size: int = 1024):
        if not is_power_of_two(block_size):
            raise ValueError(
                f"block_size must be a power of two, got {block_size}"
            )
        self.block_size = block_size

    def transform(self, vector):
        """Normalized Hadamard transform (pads or truncates to block_size)."""
        return hadamard_transform(vector, block_size=self.block_size)

    def inverse_transform(self, vector):
        """Inverse transform (identical to forward for this transform)."""
        return hadamard_transform(vector, block_size=self.block_size)

    def transform_batch(self, matrix):
        """
        Apply the transform to every row of a 2-D array at once.

        Args:
            matrix: (rows, cols) array; cols must be a power of two.

        Returns:
            (rows, block_size) array, each row transformed.
        """
        _require_numpy()
        m = np.asarray(matrix, dtype=np.float64)
        if m.ndim == 1:
            m = m[None, :]

        rows, cols = m.shape
        if cols < self.block_size:
            pad = np.zeros((rows, self.block_size - cols), dtype=np.float64)
            m = np.concatenate([m, pad], axis=1)
        elif cols > self.block_size:
            m = m[:, :self.block_size]

        out = m.copy()
        n = self.block_size
        h = 1
        while h < n:
            out = out.reshape(rows, -1, 2, h)
            a = out[:, :, 0, :].copy()
            b = out[:, :, 1, :].copy()
            out[:, :, 0, :] = a + b
            out[:, :, 1, :] = a - b
            out = out.reshape(rows, n)
            h *= 2
        return out / math.sqrt(n)

    def __repr__(self) -> str:
        return f"FastHadamardTransform(block_size={self.block_size})"

"""
Phase-encoded weights for TPN.

Weights are encoded as phase angles (radians) instead of traditional floats.
This enables:
- Continuous weight representation (not just ternary)
- Smooth fitness gradients for optimization
- Evolution through phase mutation
"""

import math
import random
from typing import List

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional
    np = None


def _require_numpy():
    """Raise a helpful error if the fast path is used without numpy."""
    if np is None:
        raise ImportError(
            "numpy is required for the vectorized fast path. "
            "Install it with: pip install numpy"
        )


def numpy_available() -> bool:
    """True when the vectorized fast path can be used."""
    return np is not None


class PhaseWeight:
    """
    A weight encoded as a phase angle.
    
    The weight value is computed as cos(phase), and the fitness is
    computed as cos²(phase - π/4).
    
    Key properties:
    - Phase 0: weight = 1.0, fitness = 0.5
    - Phase π/4: weight = 0.707, fitness = 1.0 (maximum fitness)
    - Phase π/2: weight = 0.0, fitness = 0.5
    - Phase π: weight = -1.0, fitness = 0.5
    - Phase 3π/4: weight = -0.707, fitness = 1.0 (maximum fitness)
    """
    
    def __init__(self, phase: float):
        """
        Initialize a phase weight.
        
        Args:
            phase: Phase angle in radians. Will be normalized to [0, 2π].
        """
        self.phase = phase % (2 * math.pi)
    
    @property
    def value(self) -> float:
        """Get the weight value as cos(phase)."""
        return math.cos(self.phase)
    
    @property
    def fitness(self) -> float:
        """Get the fitness as cos²(phase - π/4)."""
        return math.cos(self.phase - math.pi / 4.0) ** 2
    
    def mutate(self, mutation_rate: float = 0.1) -> 'PhaseWeight':
        """
        Create a mutated copy of this weight.
        
        Args:
            mutation_rate: Standard deviation of the Gaussian mutation.
        
        Returns:
            A new PhaseWeight with a mutated phase.
        """
        new_phase = self.phase + random.gauss(0, mutation_rate)
        return PhaseWeight(new_phase)
    
    def __repr__(self) -> str:
        return f"PhaseWeight(phase={self.phase:.4f}, value={self.value:.4f}, fitness={self.fitness:.4f})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, PhaseWeight):
            return False
        return self.phase == other.phase


class PhaseEncoding:
    """
    Utilities for encoding and decoding vectors/matrices as phase weights.
    """
    
    @staticmethod
    def encode_vector(values: List[float]) -> List[PhaseWeight]:
        """
        Encode a vector of floats as phase weights.
        
        The encoding maps each value to a phase angle such that
        cos(phase) = value. This means:
        - value 1.0 → phase 0
        - value 0.0 → phase π/2
        - value -1.0 → phase π
        
        Args:
            values: List of float values in [-1, 1].
        
        Returns:
            List of PhaseWeight objects.
        """
        encoded = []
        for v in values:
            # Clamp to [-1, 1] to avoid domain errors
            v = max(-1.0, min(1.0, v))
            phase = math.acos(v)
            encoded.append(PhaseWeight(phase))
        return encoded
    
    @staticmethod
    def decode_vector(weights: List[PhaseWeight]) -> List[float]:
        """
        Decode phase weights back to float values.
        
        Args:
            weights: List of PhaseWeight objects.
        
        Returns:
            List of float values.
        """
        return [w.value for w in weights]
    
    @staticmethod
    def encode_matrix(matrix: List[List[float]]) -> List[List[PhaseWeight]]:
        """
        Encode a matrix of floats as phase weights.
        
        Args:
            matrix: 2D list of float values.
        
        Returns:
            2D list of PhaseWeight objects.
        """
        return [PhaseEncoding.encode_vector(row) for row in matrix]
    
    @staticmethod
    def decode_matrix(weights: List[List[PhaseWeight]]) -> List[List[float]]:
        """
        Decode phase weights back to a matrix of floats.
        
        Args:
            weights: 2D list of PhaseWeight objects.
        
        Returns:
            2D list of float values.
        """
        return [PhaseEncoding.decode_vector(row) for row in weights]


class PhaseMatrix:
    """
    Vectorized phase-encoded weight matrix backed by numpy.

    This is the fast path for TPN computation. Instead of a list of
    ``PhaseWeight`` objects (one Python object per weight), the phases
    are held in a single contiguous float64 array, so ``cos`` and the
    matvec are single C-level numpy calls rather than per-element
    Python loops.

    The math is identical to the object-based path:

    - ``value``   = cos(phase)
    - ``fitness`` = cos²(phase - π/4)

    Args:
        phases: numpy array of phase angles (radians), any shape.
        copy:   if False, the array is used as-is (no defensive copy).
    """

    def __init__(self, phases, copy: bool = True, scale: float = 1.0):
        """
        Args:
            phases: phase angles (radians), any shape.
            copy:   if False, the array is used as-is (no defensive copy).
            scale:  magnitude multiplier applied on top of cos(phase).

        Because cos() only spans [-1, 1], a real weight matrix whose
        values exceed that cannot be represented by phase alone. Splitting
        the weight into ``scale * cos(phase)`` keeps the full dynamic
        range: the phase carries the *shape* of the weight and ``scale``
        carries its magnitude. See ``from_weights_auto``.
        """
        _require_numpy()
        arr = np.asarray(phases, dtype=np.float64)
        if copy:
            arr = arr.copy()
        # Normalize into [0, 2π) exactly like PhaseWeight.__init__.
        self.phases = np.mod(arr, 2.0 * math.pi)
        self.scale = float(scale)

    # -- constructors -------------------------------------------------- #

    @classmethod
    def random(cls, rows: int, cols: int, seed=None,
               center: float = math.pi / 2, spread: float = 0.1):
        """
        Random phases near ``center`` (default π/2, i.e. weight ≈ 0).

        This mirrors TPNModel._random_phase_matrix: small initial
        outputs keep the forward pass numerically stable.
        """
        _require_numpy()
        rng = np.random.default_rng(seed)
        phases = rng.normal(center, spread, size=(rows, cols))
        return cls(phases)

    @classmethod
    def from_weights(cls, weights):
        """
        Encode float weights in [-1, 1] as phases (inverse of ``values``).

        Uses acos, matching PhaseEncoding.encode_vector, but applied to
        the whole array at once. Values are clamped to the valid domain.
        """
        _require_numpy()
        arr = np.clip(np.asarray(weights, dtype=np.float64), -1.0, 1.0)
        return cls(np.arccos(arr))

    @classmethod
    def from_phase_objects(cls, matrix: List[List[PhaseWeight]]):
        """Build a PhaseMatrix from a list-of-lists of PhaseWeight."""
        _require_numpy()
        phases = [[w.phase for w in row] for row in matrix]
        return cls(phases)

    @classmethod
    def from_weights_auto(cls, weights):
        """
        Encode float weights losslessly, preserving values outside [-1, 1].

        ``cos(phase)`` can only ever produce a value in [-1, 1]. Real
        checkpoints routinely contain weights beyond that range (the
        Ling/Laguna attention projections reach ±1.22), so naively
        clamping them -- as ``PhaseEncoding.encode_vector`` does -- would
        silently flatten the largest-magnitude weights.

        Instead this splits the weight into ``scale * cos(phase)``:

            scale = max(|w|)          (0 if the matrix is all zeros)
            phase = acos(clamp(w / scale, -1, 1))

        The reconstruction ``scale * cos(phase)`` then reproduces the
        original weight matrix to float32 precision, including outliers.
        The cost is one extra scalar per matrix, not per weight.

        Args:
            weights: 2-D array of floats, any magnitude.

        Returns:
            A PhaseMatrix whose ``values()`` reconstruct the input.
        """
        _require_numpy()
        w = np.asarray(weights, dtype=np.float64)

        peak = float(np.abs(w).max()) if w.size else 0.0
        if peak == 0.0:
            # All-zero matrix: any phase reconstructs zero.
            return cls(np.zeros_like(w), scale=0.0)

        normalized = np.clip(w / peak, -1.0, 1.0)
        return cls(np.arccos(normalized), scale=peak)

    # -- properties ---------------------------------------------------- #

    @property
    def shape(self):
        return self.phases.shape

    @property
    def rows(self) -> int:
        return int(self.phases.shape[0])

    @property
    def cols(self) -> int:
        return int(self.phases.shape[1]) if self.phases.ndim > 1 else 1

    def values(self):
        """Weight values as a numpy array: scale * cos(phase)."""
        return self.scale * np.cos(self.phases)

    def fitness(self):
        """Fitness values as a numpy array: cos²(phase - π/4).

        Fitness is a property of the phase alone and is deliberately not
        scaled -- it measures how well-aligned a weight is, not its
        magnitude.
        """
        return np.cos(self.phases - math.pi / 4.0) ** 2

    # -- computation --------------------------------------------------- #

    def matvec(self, vector):
        """
        Matrix-vector product with phase-encoded weights.

        ``out[i] = sum_j scale * cos(phase[i,j]) * v[j]``

        A single BLAS call over the cached cos table, with the per-matrix
        scale folded in as one scalar multiply.
        """
        _require_numpy()
        v = np.asarray(vector, dtype=np.float64)
        return self.scale * (np.cos(self.phases) @ v)

    def matmul(self, matrix):
        """
        Matrix-matrix product: apply this weight matrix to a batch of
        column vectors at once. ``matrix`` is (cols, batch) -> (rows, batch).
        """
        _require_numpy()
        m = np.asarray(matrix, dtype=np.float64)
        return self.scale * (np.cos(self.phases) @ m)

    def to_phase_objects(self) -> List[List[PhaseWeight]]:
        """Convert back to the object-based representation."""
        return [[PhaseWeight(float(p)) for p in row] for row in self.phases]

    def tolist(self):
        """Phase angles as a nested Python list."""
        return self.phases.tolist()

    def nbytes(self) -> int:
        """Memory footprint of the phase array in bytes."""
        return int(self.phases.nbytes)

    def __repr__(self) -> str:
        return f"PhaseMatrix(shape={self.shape}, dtype={self.phases.dtype})"

"""
Full TPN Inference Engine.

A complete inference engine that uses phase-based computation for
neural network operations. This is the core of the TPN system.

Two backends are available:

- ``"numpy"`` (default when numpy is installed): phase matrices are held
  in contiguous float64 arrays and every matvec is a single BLAS call.
  Roughly three orders of magnitude faster than the object-based path.
- ``"python"``: the original list-of-``PhaseWeight`` implementation.
  Kept as the reference and for environments without numpy.

Both backends compute identical math; ``tests/test_inference_backends.py``
asserts they agree elementwise.
"""

import math
import random
from typing import List, Dict, Optional
from .phase import PhaseWeight, PhaseEncoding, PhaseMatrix, numpy_available
from .attention import TPNAttention, tpn_softmax
from .hadamard import HadamardTransform
from .evolution import EvolvablePhaseNetwork

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional
    np = None


# Weight names in a transformer block, in the order they are applied.
BLOCK_WEIGHT_NAMES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def _silu(x: float) -> float:
    """SiLU activation: x * sigmoid(x), numerically stable."""
    if x >= 0:
        return x / (1.0 + math.exp(-x))
    ex = math.exp(x)
    return x * ex / (1.0 + ex)


class TPNConfig:
    """Configuration for TPN model."""

    def __init__(self, hidden_size: int = 512, num_layers: int = 4,
                 num_heads: int = 4, head_dim: int = 128,
                 intermediate_size: int = 2048, max_position_embeddings: int = 2048,
                 vocab_size: int = 32000):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size

    def to_dict(self) -> Dict:
        """Configuration as a plain dictionary."""
        return {
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "head_dim": self.head_dim,
            "intermediate_size": self.intermediate_size,
            "max_position_embeddings": self.max_position_embeddings,
            "vocab_size": self.vocab_size,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TPNConfig':
        """Build a config from a dictionary (unknown keys are ignored)."""
        known = {
            "hidden_size", "num_layers", "num_heads", "head_dim",
            "intermediate_size", "max_position_embeddings", "vocab_size",
        }
        return cls(**{k: v for k, v in data.items() if k in known})


class TPNModel:
    """
    TPN Inference Model.

    A neural network that uses phase-based computation for all operations:
    - Phase-encoded weights
    - Fitness-based attention
    - Hadamard transform for efficient computation
    - Evolution for weight optimization

    Args:
        config: TPNConfig describing the architecture.
        backend: "auto" (default), "numpy" or "python". "auto" selects
            numpy when it is importable, otherwise the pure-Python path.
        seed: optional seed for reproducible weight initialization.
    """

    def __init__(self, config: TPNConfig, backend: str = "auto", seed=None):
        if backend == "auto":
            backend = "numpy" if numpy_available() else "python"
        if backend == "numpy" and not numpy_available():
            raise ImportError(
                "backend='numpy' requires numpy. Install it with: "
                "pip install numpy"
            )
        if backend not in ("numpy", "python"):
            raise ValueError(f"unknown backend {backend!r}")

        self.config = config
        self.backend = backend
        self.hadamard = HadamardTransform(block_size=1024)

        # Dedicated RNG for reproducible initialization. Using a local
        # generator keeps model init from perturbing the global random
        # state that evolution/trajectory code relies on.
        self._rng = random.Random(seed)

        self.layers = self._init_layers()

    # ------------------------------------------------------------------ #
    # Initialization
    # ------------------------------------------------------------------ #

    def _init_layers(self) -> List[Dict]:
        """Initialize model layers with random phase-encoded weights."""
        c = self.config
        layers = []
        for i in range(c.num_layers):
            layer = {
                "index": i,
                "q_proj": self._random_phase_matrix(c.hidden_size, c.hidden_size),
                "k_proj": self._random_phase_matrix(c.hidden_size, c.hidden_size),
                "v_proj": self._random_phase_matrix(c.hidden_size, c.hidden_size),
                "o_proj": self._random_phase_matrix(c.hidden_size, c.hidden_size),
                "gate_proj": self._random_phase_matrix(c.intermediate_size, c.hidden_size),
                "up_proj": self._random_phase_matrix(c.intermediate_size, c.hidden_size),
                "down_proj": self._random_phase_matrix(c.hidden_size, c.intermediate_size),
            }
            layers.append(layer)
        return layers

    def _random_phase_matrix(self, rows: int, cols: int):
        """
        Create a matrix of random phase-encoded weights with small values.

        Phases are centered on pi/2 (weight ~ 0) with a small spread, which
        keeps initial outputs small and the forward pass stable.

        The numpy path uses a vectorized draw (rows * cols in a single
        numpy call) instead of one Python random per cell, so building a
        large model is fast. The python path keeps the exact per-cell
        behaviour of the original engine for reference.
        """
        if self.backend == "numpy":
            rng = np.random.default_rng(self._rng.randint(0, 2**31))
            phases = rng.normal(
                math.pi / 2, 0.1, size=(rows, cols),
            )
            return PhaseMatrix(phases, copy=False)
        return [[PhaseWeight(math.pi / 2 + self._rng.gauss(0, 0.1))
                 for _ in range(cols)] for _ in range(rows)]

    # ------------------------------------------------------------------ #
    # Primitives (backend-aware)
    # ------------------------------------------------------------------ #

    def _normalize(self, vector):
        """Normalize vector to unit length (near-zero vectors pass through)."""
        if self.backend == "numpy":
            v = np.asarray(vector, dtype=np.float64)
            norm = float(np.sqrt(np.dot(v, v)))
            if norm > 1e-6:
                return v / norm
            return v
        norm = math.sqrt(sum(x * x for x in vector))
        if norm > 1e-6:
            return [x / norm for x in vector]
        return vector

    def _clamp(self, vector, max_val: float = 10.0):
        """Clamp vector values to prevent explosion."""
        if self.backend == "numpy":
            return np.clip(np.asarray(vector, dtype=np.float64),
                           -max_val, max_val)
        return [max(-max_val, min(max_val, x)) for x in vector]

    def _matvec(self, matrix, vector):
        """Matrix-vector product with phase-encoded weights."""
        if isinstance(matrix, PhaseMatrix):
            return matrix.matvec(vector)
        result = [0.0] * len(matrix)
        for i in range(len(matrix)):
            s = 0.0
            row = matrix[i]
            for j in range(len(vector)):
                s += row[j].value * vector[j]
            result[i] = s
        return result

    def _matvec_batch(self, matrix, vectors):
        """
        Apply one weight matrix to several vectors in a single traversal.

        ``vectors`` is a (batch, cols) array; returns (batch, rows).
        This is the weight-stationary batching the bitsliced path is
        built on: the weight matrix is read once for the whole batch.
        """
        if isinstance(matrix, PhaseMatrix):
            m = np.asarray(vectors, dtype=np.float64)
            # (rows, cols) @ (cols, batch) -> (rows, batch), then transpose
            return matrix.scale * (np.cos(matrix.phases) @ m.T).T
        return [self._matvec(matrix, v) for v in vectors]

    # ------------------------------------------------------------------ #
    # Forward pass
    # ------------------------------------------------------------------ #

    def _attention(self, q, k, v):
        """
        Single-vector attention using the TPN fitness softmax.

        The score is the scaled dot product of q and k; the output is
        that score's fitness times v. Identical math in both backends.
        """
        if self.backend == "numpy":
            d = len(q)
            score = float(np.dot(q, k)) / math.sqrt(d)
            fitness = math.cos(score - math.pi / 4.0) ** 2
            return fitness * v
        d = len(q)
        score = sum(q[i] * k[i] for i in range(d)) / math.sqrt(d)
        fitness = math.cos(score - math.pi / 4.0) ** 2
        return [fitness * v[i] for i in range(len(v))]

    def _ffn(self, x, layer):
        """Feed-forward network with phase-encoded weights."""
        gate = self._matvec(layer["gate_proj"], x)
        up = self._matvec(layer["up_proj"], x)

        if self.backend == "numpy":
            # SiLU: x * sigmoid(x), computed elementwise in numpy.
            ffn = (gate / (1.0 + np.exp(-gate))) * up
        else:
            ffn = [_silu(g) * u for g, u in zip(gate, up)]

        return self._matvec(layer["down_proj"], ffn)

    def forward(self, input_data) -> List[float]:
        """
        Forward pass through the TPN model.

        Args:
            input_data: Input vector of size hidden_size.

        Returns:
            Output vector of size hidden_size.
        """
        hidden_size = self.config.hidden_size

        if self.backend == "numpy":
            hidden = self._normalize(
                np.asarray(input_data, dtype=np.float64)[:hidden_size]
            )
            for layer in self.layers:
                q = self._normalize(self._matvec(layer["q_proj"], hidden))
                k = self._normalize(self._matvec(layer["k_proj"], hidden))
                v = self._normalize(self._matvec(layer["v_proj"], hidden))

                attn_out = self._attention(q, k, v)
                hidden = self._normalize(
                    hidden + self._matvec(layer["o_proj"], attn_out)
                )

                ffn_out = self._ffn(hidden, layer)
                hidden = self._normalize(hidden + ffn_out)
            return hidden.tolist()

        hidden = self._normalize(input_data[:hidden_size])
        for layer in self.layers:
            q = self._normalize(self._matvec(layer["q_proj"], hidden))
            k = self._normalize(self._matvec(layer["k_proj"], hidden))
            v = self._normalize(self._matvec(layer["v_proj"], hidden))

            attn_out = self._attention(q, k, v)
            hidden = self._normalize(
                [h + a for h, a in zip(hidden,
                                       self._matvec(layer["o_proj"], attn_out))]
            )

            ffn_out = self._ffn(hidden, layer)
            hidden = self._normalize([h + f for h, f in zip(hidden, ffn_out)])

        return hidden

    def forward_batch(self, inputs) -> List[List[float]]:
        """
        Forward pass over a batch of inputs, sharing each weight traversal.

        The weight matrices are read once per layer for the whole batch
        rather than once per input. The per-input math is identical to
        ``forward``; ``tests/test_inference_backends.py`` asserts the two
        agree.

        Args:
            inputs: sequence of input vectors.

        Returns:
            List of output vectors, one per input.
        """
        if self.backend != "numpy":
            return [self.forward(x) for x in inputs]

        hidden_size = self.config.hidden_size
        if len(inputs) == 0:
            return []

        h = np.asarray(
            [np.asarray(x, dtype=np.float64)[:hidden_size] for x in inputs],
            dtype=np.float64,
        )
        h = self._normalize_batch(h)

        for layer in self.layers:
            q = self._normalize_batch(self._matvec_batch(layer["q_proj"], h))
            k = self._normalize_batch(self._matvec_batch(layer["k_proj"], h))
            v = self._normalize_batch(self._matvec_batch(layer["v_proj"], h))

            # scores[b] = <q_b, k_b> / sqrt(d); fitness applied elementwise
            d = q.shape[1]
            scores = np.sum(q * k, axis=1) / math.sqrt(d)
            fitness = np.cos(scores - math.pi / 4.0) ** 2
            attn_out = fitness[:, None] * v

            h = self._normalize_batch(
                h + self._matvec_batch(layer["o_proj"], attn_out)
            )

            gate = self._matvec_batch(layer["gate_proj"], h)
            up = self._matvec_batch(layer["up_proj"], h)
            ffn = (gate / (1.0 + np.exp(-gate))) * up
            h = self._normalize_batch(
                h + self._matvec_batch(layer["down_proj"], ffn)
            )

        return h.tolist()

    def _normalize_batch(self, matrix):
        """Row-wise L2 normalization; near-zero rows are left unchanged."""
        norms = np.sqrt(np.sum(matrix * matrix, axis=1, keepdims=True))
        safe = np.where(norms > 1e-6, norms, 1.0)
        return matrix / safe

    # ------------------------------------------------------------------ #
    # Weight loading
    # ------------------------------------------------------------------ #

    def weight_shapes(self) -> Dict[str, tuple]:
        """Expected shape of every weight matrix in a block."""
        c = self.config
        return {
            "q_proj": (c.hidden_size, c.hidden_size),
            "k_proj": (c.hidden_size, c.hidden_size),
            "v_proj": (c.hidden_size, c.hidden_size),
            "o_proj": (c.hidden_size, c.hidden_size),
            "gate_proj": (c.intermediate_size, c.hidden_size),
            "up_proj": (c.intermediate_size, c.hidden_size),
            "down_proj": (c.hidden_size, c.intermediate_size),
        }

    def set_weight_matrix(self, layer_idx: int, name: str, weights,
                          encode: bool = True, preserve_scale: bool = True) -> None:
        """
        Install a weight matrix into a layer from float weights.

        Args:
            layer_idx: which block to write into.
            name: one of BLOCK_WEIGHT_NAMES ("q_proj", "gate_proj", ...).
            weights: 2-D array of floats, or -- when ``encode`` is False --
                raw phase angles in radians.
            encode: when True (default) interpret ``weights`` as weight
                values and encode them as phases via arccos.
            preserve_scale: when True (default), weights whose magnitude
                exceeds 1 are represented losslessly as
                ``scale * cos(phase)`` instead of being clamped to ±1.
                Set False only if you specifically want the hard clamp.
        """
        if name not in BLOCK_WEIGHT_NAMES:
            raise ValueError(
                f"unknown weight name {name!r}; expected one of "
                f"{BLOCK_WEIGHT_NAMES}"
            )
        if not 0 <= layer_idx < len(self.layers):
            raise IndexError(
                f"layer_idx {layer_idx} out of range "
                f"(model has {len(self.layers)} layers)"
            )

        layer = self.layers[layer_idx]

        if self.backend == "numpy":
            arr = np.asarray(weights, dtype=np.float64)
            expected = tuple(layer[name].shape)
            if arr.shape != expected:
                raise ValueError(
                    f"{name} for layer {layer_idx} expects shape {expected}, "
                    f"got {arr.shape}"
                )
            if not encode:
                layer[name] = PhaseMatrix(arr)
            elif preserve_scale:
                layer[name] = PhaseMatrix.from_weights_auto(arr)
            else:
                layer[name] = PhaseMatrix.from_weights(arr)
            return

        rows = [[float(v) for v in row] for row in weights]
        expected_shape = (len(layer[name]), len(layer[name][0]))
        got = (len(rows), len(rows[0]) if rows else 0)
        if got != expected_shape:
            raise ValueError(
                f"{name} for layer {layer_idx} expects shape {expected_shape}, "
                f"got {got}"
            )
        if encode:
            layer[name] = PhaseEncoding.encode_matrix(rows)
        else:
            layer[name] = [[PhaseWeight(v) for v in row] for row in rows]

    def load_block(self, layer_idx: int, weights: Dict, encode: bool = True,
                   strict: bool = False) -> Dict[str, str]:
        """
        Install several weight matrices into one layer at once.

        Args:
            layer_idx: which block to write into.
            weights: mapping of weight name -> 2-D array. Names outside
                BLOCK_WEIGHT_NAMES are ignored.
            encode: interpret arrays as weight values (True) or raw
                phases (False).
            strict: when True, a shape mismatch raises; when False the
                offending matrix is skipped and reported.

        Returns:
            Mapping of name -> "loaded" or a short reason it was skipped.
        """
        report: Dict[str, str] = {}
        for name, arr in weights.items():
            if name not in BLOCK_WEIGHT_NAMES:
                continue
            try:
                self.set_weight_matrix(layer_idx, name, arr, encode=encode)
                report[name] = "loaded"
            except (ValueError, IndexError) as exc:
                if strict:
                    raise
                report[name] = f"skipped: {exc}"
        return report

    # ------------------------------------------------------------------ #
    # Introspection
    # ------------------------------------------------------------------ #

    def num_parameters(self) -> int:
        """Total number of weights in the model."""
        c = self.config
        per_layer = (
            4 * c.hidden_size * c.hidden_size          # q, k, v, o
            + 3 * c.intermediate_size * c.hidden_size  # gate, up, down
        )
        return per_layer * c.num_layers

    def memory_bytes(self) -> int:
        """Footprint of the phase weights, in bytes."""
        if self.backend == "numpy":
            return sum(
                layer[name].nbytes()
                for layer in self.layers
                for name in BLOCK_WEIGHT_NAMES
            )
        # Object path: one Python object per weight plus its float.
        import sys
        obj_bytes = sys.getsizeof(PhaseWeight(0.0)) + 8
        return self.num_parameters() * obj_bytes

    # ------------------------------------------------------------------ #
    # Evolution
    # ------------------------------------------------------------------ #

    def evolve(self, target: List[float], generations: int = 50,
               mutation_rate: float = 0.1) -> Dict:
        """
        Evolve the model to match target output.

        Args:
            target: Target output vector.
            generations: Number of generations to evolve.
            mutation_rate: Standard deviation of Gaussian mutation.

        Returns:
            Dictionary with evolution results.
        """
        # Create evolvable network for each layer
        networks = []
        for layer in self.layers:
            network = EvolvablePhaseNetwork(num_weights=self.config.hidden_size)
            networks.append(network)

        # Evolve each layer
        results = []
        for i, network in enumerate(networks):
            result = network.evolve(target, mutation_rate=mutation_rate,
                                    generations=generations)
            results.append(result)

        return {
            "generations": generations,
            "final_fitness": results[-1]["final_fitness"],
            "layer_results": results,
        }

    def get_config(self) -> Dict:
        """Get model configuration."""
        return self.config.to_dict()

    def __repr__(self) -> str:
        c = self.config
        return (f"TPNModel(hidden_size={c.hidden_size}, layers={c.num_layers}, "
                f"intermediate_size={c.intermediate_size}, "
                f"backend={self.backend!r})")

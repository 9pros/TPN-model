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
from typing import Any, Dict, List, Optional, Tuple, Union
from .phase import PhaseWeight, PhaseEncoding, PhaseMatrix, numpy_available
from .attention import TPNAttention, tpn_softmax
from .hadamard import HadamardTransform
from .evolution import EvolvablePhaseNetwork

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is optional
    np = None


# --------------------------------------------------------------------- #
# Pure-Python sampling / vector helpers (used by TPNModel.generate;
# they work whether or not numpy is installed).
# --------------------------------------------------------------------- #


def _vec_add_scale(a: List[float], b: List[float], scale: float) -> List[float]:
    """Elementwise a + scale * b."""
    return [ai + scale * bi for ai, bi in zip(a, b)]


def _normalize_vec(x: List[float]) -> List[float]:
    """L2-normalize a plain list of floats."""
    norm = math.sqrt(sum(v * v for v in x)) or 1.0
    return [v / norm for v in x]


def _weighted_sample_simple(rng: random.Random,
                            probs: List[float]) -> int:
    """Weighted random choice over a probability list."""
    r = rng.random()
    cum = 0.0
    for i, p in enumerate(probs):
        cum += p
        if r <= cum:
            return i
    return len(probs) - 1


def _top_k_sample(rng: random.Random,
                  probs: List[float], k: int) -> int:
    """Sample from the top-k largest probabilities."""
    if k <= 0 or k >= len(probs):
        return _weighted_sample_simple(rng, probs)
    top = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)[:k]
    w = [probs[i] / sum(probs[j] for j in top) for i in top]
    r = rng.random()
    cum = 0.0
    for i, wi in zip(top, w):
        cum += wi
        if r <= cum:
            return i
    return top[-1]


def _top_p_sample(rng: random.Random,
                  probs: List[float], p: float) -> int:
    """Sample from the smallest set of entries whose cumulative prob >= p."""
    if not (0.0 < p <= 1.0):
        return _weighted_sample_simple(rng, probs)
    idx = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    cum = 0.0
    chosen = []
    for i in idx:
        cum += probs[i]
        chosen.append(i)
        if cum >= p:
            break
    tot = cum
    r = rng.random()
    acc = 0.0
    for i in chosen:
        acc += probs[i] / tot
        if r <= acc:
            return i
    return chosen[-1]


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
        self.word_embeddings = None   # root matrix: vocab x hidden
        self.lm_head = None           # root matrix: vocab x hidden
        self._init_root_embeddings()
        self.generation_history: List[Dict] = []  # evolution/provenance log

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
               mutation_rate: float = 0.1, register_provenance: bool = True) -> Dict:
        """
        Evolve the model to match target output.

        Args:
            target: Target output vector.
            generations: Number of generations to evolve.
            mutation_rate: Standard deviation of Gaussian mutation.
            register_provenance: Record every generation into the model's
                Binetic identity for provenance/rollback (if the model is
                Binetic-authored).

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
        best_fitness_history = []
        for i, network in enumerate(networks):
            result = network.evolve(target, mutation_rate=mutation_rate,
                                    generations=generations)
            results.append(result)
            best_fitness_history.append(result["final_fitness"])

        final_fitness = best_fitness_history[-1]

        # Log this evolution run to the model's provenance history.
        self.generation_history.append({
            "type": "evolve",
            "generations": generations,
            "final_fitness": final_fitness,
            "mutation_rate": mutation_rate,
            "target_length": len(target),
        })

        # --- Binetic provenance: record the evolution run ---
        if register_provenance and self.binetic_identity is not None:
            from .binetic import GenerationRecord
            graph = getattr(self, "binetic_graph", None)
            if graph is None:
                graph = self.to_binetic_graph()
                self.binetic_graph = graph
            for gen in range(generations):
                fitness = best_fitness_history[gen]
                record = GenerationRecord(
                    generation=gen,
                    fitness=fitness,
                    mutation_rate=mutation_rate,
                    target_hash=str(abs(hash(tuple(target)) & 0xFFFFFFFF)),
                )
                graph.register_generation(record)
            # Update identity generation info
            self.binetic_identity.generation_info = {
                **self.binetic_identity.generation_info,
                "evolved": True,
                "target_length": len(target),
                "best_generation_fitness": max(best_fitness_history),
            }

        return {
            "generations": generations,
            "final_fitness": final_fitness,
            "layer_results": results,
            "best_fitness_history": best_fitness_history,
        }

    def get_config(self) -> Dict:
        """Get model configuration."""
        return self.config.to_dict()

    def _init_root_embeddings(self):
        """Initialize optional root matrices if a loader didn't provide them."""
        c = self.config
        if self.word_embeddings is None:
            self.word_embeddings = self._random_root(c.hidden_size,
                                                     c.vocab_size)
        if self.lm_head is None:
            self.lm_head = self._random_root(c.hidden_size, c.vocab_size)

    def _random_root(self, rows: int, cols: int):
        """Random root matrix, phases centered near zero for stability."""
        if self.backend == "numpy":
            rng = np.random.default_rng(self._rng.randint(0, 2**31))
            phases = rng.normal(math.pi / 2, 0.1, size=(rows, cols))
            return PhaseMatrix(phases)
        return [[PhaseWeight(self._rng.gauss(math.pi / 2, 0.1))
                 for _ in range(cols)] for _ in range(rows)]

    def set_root_matrix(self, name: str, weights,
                        encode: bool = True,
                        preserve_scale: bool = True) -> None:
        """
        Install a root (non-block) weight matrix used for generation.

        Args:
            name: one of ``word_embeddings`` or ``lm_head``.
            weights: 2-D array of floats, or raw phases when ``encode=False``.
            encode: interpret ``weights`` as weight values or raw phases.
            preserve_scale: lossless encoding for |w| > 1.

        Raises:
            ValueError: if ``name`` is not a known root matrix.
        """
        if name == "word_embeddings":
            target = self.word_embeddings
            if target is None:
                target = self._random_root(self.config.vocab_size,
                                           self.config.hidden_size)
            if self.backend == "numpy":
                arr = np.asarray(weights, dtype=np.float64)
                if not encode:
                    self.word_embeddings = PhaseMatrix(arr)
                elif preserve_scale:
                    self.word_embeddings = PhaseMatrix.from_weights_auto(arr)
                else:
                    self.word_embeddings = PhaseMatrix.from_weights(arr)
            else:
                rows = [[float(v) for v in row] for row in weights]
                if encode:
                    self.word_embeddings = PhaseEncoding.encode_matrix(rows)
                else:
                    self.word_embeddings = [[PhaseWeight(v) for v in row]
                                            for row in rows]
        elif name == "lm_head":
            target = self.lm_head
            if target is None:
                target = self._random_root(self.config.vocab_size,
                                           self.config.hidden_size)
            if self.backend == "numpy":
                arr = np.asarray(weights, dtype=np.float64)
                if not encode:
                    self.lm_head = PhaseMatrix(arr)
                elif preserve_scale:
                    self.lm_head = PhaseMatrix.from_weights_auto(arr)
                else:
                    self.lm_head = PhaseMatrix.from_weights(arr)
            else:
                rows = [[float(v) for v in row] for row in weights]
                if encode:
                    self.lm_head = PhaseEncoding.encode_matrix(rows)
                else:
                    self.lm_head = [[PhaseWeight(v) for v in row]
                                    for row in rows]
        else:
            raise ValueError(f"unknown root weight name {name!r}")

    def _embed_token(self, token_id: int) -> Union[List[float], np.ndarray]:
        """Convert a token id into a hidden-size input vector."""
        h = self.config.hidden_size
        if self.word_embeddings is None:
            # Demo fallback: one-hot over a bounded index.
            x = [0.0] * h
            x[min(token_id, h - 1)] = 1.0
            return x
        if self.backend == "numpy" and isinstance(self.word_embeddings,
                                                   PhaseMatrix):
            if not hasattr(self, "_word_embeddings_values"):
                self._word_embeddings_values = np.asarray(
                    self.word_embeddings.values())
            return self._word_embeddings_values[token_id]
        if isinstance(self.word_embeddings, PhaseMatrix):
            if not hasattr(self, "_word_embeddings_values"):
                self._word_embeddings_values = np.asarray(
                    self.word_embeddings.values())
            return [float(v) for v in self._word_embeddings_values[token_id]]
        return [w.value for w in self.word_embeddings[token_id]]

    def _lm_head_logits(self, hidden: Union[List[float], Any]) -> Union[List[float], Any]:
        """Compute logits over the vocabulary from a hidden state."""
        if self.lm_head is None:
            return [0.0] * self.config.vocab_size
        if self.backend == "numpy":
            return self.lm_head.matvec(hidden)
        return [self._matvec(self.lm_head[i], hidden)
                for i in range(len(self.lm_head))]

    # ------------------------------------------------------------------ #
    # Generation with temporal context
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        tokenizer,
        max_tokens: int = 20,
        context=None,
        blend_memory: bool = False,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_tokens: int = 0,
        verbose: bool = False,
        return_tokens: bool = False,
        seed: Optional[int] = None,
    ) -> Tuple[str, Optional[List[int]]]:
        """
        Autoregressively generate text from ``prompt``.

        Statefulness via temporal context: if a ``context`` is provided,
        each token's hidden state is written into it and, optionally,
        blended into the next token's input, so the model remembers the
        prefix across calls. The context can be persisted with
        ``save_context`` / ``load_context`` and reused.

        Args:
            prompt: Input text.
            tokenizer: A ``Tokenizer`` with ``encode`` / ``decode``.
            max_tokens: Number of tokens to generate.
            context: TemporalContext for stateful generation.
            blend_memory: Blend retrieved context into each new token.
            temperature: Sampling temperature.
            top_k: Optional top-k sampling limit.
            top_p: Optional nucleus (top-p) sampling limit.
            min_tokens: Minimum tokens to emit before stopping.
            verbose: Print progress per step.
            return_tokens: Also return the generated token ids.
            seed: Optional seed for reproducible sampling.

        Returns:
            Generated text, optionally with ``(text, token_ids)``.
        """
        self._init_root_embeddings()

        rng = random.Random(seed)
        token_ids = tokenizer.encode(prompt, add_eos=False,
                                     max_length=self.config.max_position_embeddings)
        t = context.length if context is not None else 0
        generated = []

        # Process the prompt (feed into the model and into context).
        for token_id in token_ids:
            x = self._embed_token(token_id)
            hidden = self.forward(x)
            if context is not None:
                context.update(hidden.tolist(), t)
                t += 1

        t_start = t
        while len(generated) < max_tokens:
            last_id = token_ids[-1] if not generated else generated[-1]
            x = self._embed_token(last_id)
            if context is not None and blend_memory and context.length > 0:
                mem = context.retrieve(t)
                if numpy_available():
                    x = x + 0.5 * np.asarray(mem, dtype=np.float64)
                    x = self._normalize(x)
                else:
                    x = _vec_add_scale(x, mem, 0.5)
                    x = _normalize_vec(x)
            hidden = self.forward(x)
            if context is not None:
                context.update(hidden.tolist(), t)
            logits = self._lm_head_logits(hidden)
            m = float(np.max(logits)) if numpy_available() else max(logits)
            exps = [math.exp(v - m) for v in logits]
            probs = [e / sum(exps) for e in exps]

            if top_k is not None and top_k > 0:
                token_id = _top_k_sample(rng, probs, top_k)
            elif top_p is not None:
                token_id = _top_p_sample(rng, probs, top_p)
            else:
                token_id = _weighted_sample_simple(rng, probs)

            generated.append(token_id)
            t += 1
            if verbose:
                print(f"[{t - t_start}] "
                      + tokenizer.decode([token_id])[:30])
            if token_id == tokenizer.eos_token_id:
                generated.pop()
                break
            if len(generated) >= min_tokens:
                break

        text = tokenizer.decode(generated, skip_special=True)
        if return_tokens:
            return text, generated
        return text

    # ------------------------------------------------------------------ #
    # Temporal context persistence helpers
    # ------------------------------------------------------------------ #

    def save_context(self, path: str, context) -> None:
        """Persist a temporal context to disk (alias for ``context.save``)."""
        context.save(path)

    def load_context(self, path: str):
        """Load a temporal context from a persisted file."""
        return TemporalContext.load(path)

    def num_parameters(self) -> int:
        """Total number of layer weights in the model (layers only)."""
        c = self.config
        per_layer = (
            4 * c.hidden_size * c.hidden_size          # q, k, v, o
            + 3 * c.intermediate_size * c.hidden_size  # gate, up, down
        )
        return per_layer * c.num_layers

    def num_parameters_with_roots(self) -> int:
        """Total weights including root embeddings (word_embeddings/lm_head)."""
        total = self.num_parameters()
        if self.word_embeddings is not None:
            total += self._matrix_size(self.word_embeddings)
        if self.lm_head is not None:
            total += self._matrix_size(self.lm_head)
        return total

    @staticmethod
    def _matrix_size(matrix) -> int:
        """Count the number of scalar weights in a weight matrix."""
        if isinstance(matrix, PhaseMatrix):
            return matrix.nbytes() // 8  # float64 phases
        if isinstance(matrix, list) and matrix:
            return len(matrix) * len(matrix[0])
        return 0

    # ------------------------------------------------------------------ #
    # Binetic persistence: TPNModel <-> .binetic with identity
    # ------------------------------------------------------------------ #

    def to_binetic_graph(self) -> "BineticGraph":
        """Convert this model into a Binetic graph (for provenance/saving)."""
        from .binetic import BineticGraph, BineticIdentity

        def _serializable(weights):
            """Convert a weight matrix to plain JSON-serializable data."""
            if isinstance(weights, PhaseMatrix):
                return [[float(w.phase) for w in row]
                        for row in weights.to_phase_objects()]
            if isinstance(weights, list) and weights:
                if isinstance(weights[0], PhaseWeight):
                    return [[float(w.phase) for w in row]]
                if isinstance(weights[0], list):
                    return weights
            return weights

        graph = BineticGraph()
        graph.set_binetic_identity(
            owner="binetic.ai",
            partners=["Max Yeremenko", "binetic-partner"],
            tool="tpn_model_export",
            derived_from="TPNModel",
            generation_info={
                "model_type": "TPNModel",
                "hidden_size": self.config.hidden_size,
                "num_layers": self.config.num_layers,
                "intermediate_size": self.config.intermediate_size,
                "vocab_size": self.config.vocab_size,
                "backend": self.backend,
                "num_parameters": self.num_parameters_with_roots(),
                "evolution_history": self.generation_history,
                "binetic_generations": (
                    graph.metadata.get("binetic_generations", [])
                    if "graph" in locals() else []),
            },
        )
        layers = []
        for i, layer in enumerate(self.layers):
            node_data = {"type": "transformer_block", "index": i,
                         "layer_name": f"model.layers.{i}"}
            for weight_key in [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"]:
                w = layer.get(weight_key)
                if w is not None:
                    node_data[f"{weight_key}_phase_weights"] = (
                        _serializable(w))
            layers.append({"name": f"layer_{i}", "type": "transformer",
                           "weights": node_data})
            graph.add_node(f"layer_{i}", node_data)
            if i > 0:
                graph.add_edge(f"layer_{i-1}", f"layer_{i}",
                               {"type": "sequential"})
        graph.metadata["model_type"] = "TPNModel"
        return graph

    def save_to_binetic(self, filepath: str, version_id: str = "v1",
                        inference_id: Optional[str] = None,
                        auto_save: bool = True) -> None:
        """
        Save this model to ``.binetic`` format.

        The saved file carries the Binetic identity (owner binetic.ai,
        partners Max Yeremenko & binetic-partner), so any loader can prove
        this model was authored by Binetic.
        """
        from .binetic import BineticFormat, BineticIdentity
        fmt = BineticFormat()
        fmt.set_default_identity(
            BineticIdentity(owner="binetic.ai",
                            partners=["Max Yeremenko", "binetic-partner"],
                            tool="tpn_model_save"))
        graph = self.to_binetic_graph()
        fmt.save(filepath, graph, version_id=version_id,
                 inference_id=inference_id)
        if auto_save:
            fmt.auto_save(filepath, graph, inference_id or "v1")

    def load_from_binetic(self, filepath: str) -> "BineticGraph":
        """
        Load a model from ``.binetic`` format.

        On success the loaded graph is attached to the model so the model
        remembers where it came from: ``model.binetic_graph`` and
        ``model.binetic_identity``.
        """
        from .binetic import BineticFormat
        fmt = BineticFormat()
        graph, version = fmt.load(filepath)
        self.binetic_graph = graph
        self.binetic_identity = graph.get_binetic_identity()
        return graph

    @property
    def binetic_identity(self) -> Optional["BineticIdentity"]:
        """The Binetic identity this model carries, if any."""
        return getattr(self, "_binetic_identity", None)

    @binetic_identity.setter
    def binetic_identity(
            self, value: Optional["BineticIdentity"]) -> None:
        self._binetic_identity = value

    def is_binetic(self) -> bool:
        """True if this model was authored/trained/converted by Binetic."""
        return (self.binetic_identity is not None
                and self.binetic_identity.is_binetic)

    def __repr__(self) -> str:
        c = self.config
        return (f"TPNModel(hidden_size={c.hidden_size}, layers={c.num_layers}, "
                f"intermediate_size={c.intermediate_size}, "
                f"backend={self.backend!r})")

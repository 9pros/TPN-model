"""
Equivalence tests for the vectorized (numpy) fast paths.

The optimization work must not change any math. Every fast path is
asserted elementwise against the reference implementation it replaces,
so a speedup can never be bought with a silent correctness regression.

Covered:
- PhaseMatrix vs PhaseWeight / PhaseEncoding
- attention_fast vs attention
- hadamard_fast vs hadamard (matrix implementation)
- TPNModel numpy backend vs python backend
- forward_batch vs repeated forward
"""

import math
import random
import pytest

np = pytest.importorskip("numpy")

from tpn_engine.phase import (
    PhaseWeight, PhaseEncoding, PhaseMatrix, numpy_available,
)
from tpn_engine.attention import (
    TPNAttention, tpn_softmax, standard_softmax,
)
from tpn_engine.attention_fast import (
    TPNAttentionFast, tpn_softmax_fast, standard_softmax_fast,
)
from tpn_engine.hadamard import HadamardTransform
from tpn_engine.hadamard_fast import (
    FastHadamardTransform, fwht, hadamard_transform, next_power_of_two,
    is_power_of_two,
)
from tpn_engine.inference import TPNModel, TPNConfig

TOL = 1e-10


# --------------------------------------------------------------------- #
# PhaseMatrix
# --------------------------------------------------------------------- #

class TestPhaseMatrixEquivalence:
    def test_values_match_phase_objects(self):
        rng = random.Random(0)
        objects = [[PhaseWeight(rng.uniform(0, 2 * math.pi)) for _ in range(6)]
                   for _ in range(4)]
        matrix = PhaseMatrix.from_phase_objects(objects)

        ref = [[w.value for w in row] for row in objects]
        got = matrix.values()
        assert np.allclose(ref, got, atol=TOL)

    def test_fitness_matches_phase_objects(self):
        rng = random.Random(1)
        objects = [[PhaseWeight(rng.uniform(0, 2 * math.pi)) for _ in range(6)]
                   for _ in range(4)]
        matrix = PhaseMatrix.from_phase_objects(objects)

        ref = [[w.fitness for w in row] for row in objects]
        got = matrix.fitness()
        assert np.allclose(ref, got, atol=TOL)

    def test_matvec_matches_object_matvec(self):
        rng = random.Random(2)
        for _ in range(25):
            rows, cols = rng.randint(1, 10), rng.randint(1, 10)
            objects = [[PhaseWeight(rng.uniform(0, 2 * math.pi))
                        for _ in range(cols)] for _ in range(rows)]
            matrix = PhaseMatrix.from_phase_objects(objects)
            v = [rng.uniform(-2, 2) for _ in range(cols)]

            ref = [sum(objects[i][j].value * v[j] for j in range(cols))
                   for i in range(rows)]
            got = matrix.matvec(v)
            assert np.allclose(ref, got, atol=TOL)

    def test_from_weights_inverts_values(self):
        """PhaseMatrix.from_weights must round-trip through values()."""
        rng = random.Random(3)
        weights = [[rng.uniform(-1, 1) for _ in range(8)] for _ in range(5)]
        matrix = PhaseMatrix.from_weights(weights)
        assert np.allclose(matrix.values(), weights, atol=TOL)

    def test_from_weights_matches_phase_encoding(self):
        rng = random.Random(4)
        weights = [rng.uniform(-1, 1) for _ in range(20)]
        encoded = PhaseEncoding.encode_vector(weights)
        matrix = PhaseMatrix.from_weights(np.array(weights).reshape(1, -1))

        ref = [w.value for w in encoded]
        assert np.allclose(matrix.values()[0], ref, atol=TOL)

    def test_phases_normalized_to_2pi(self):
        matrix = PhaseMatrix([-1.0, 3 * math.pi, 7 * math.pi])
        assert np.all(matrix.phases >= 0.0)
        assert np.all(matrix.phases < 2 * math.pi + 1e-12)

    def test_matmul_matches_per_vector_matvec(self):
        rng = random.Random(5)
        objects = [[PhaseWeight(rng.uniform(0, 2 * math.pi)) for _ in range(6)]
                   for _ in range(4)]
        matrix = PhaseMatrix.from_phase_objects(objects)
        batch = [[rng.uniform(-2, 2) for _ in range(6)] for _ in range(3)]

        # matmul takes (cols, batch)
        as_cols = np.array(batch).T
        got = matrix.matmul(as_cols).T
        ref = np.array([matrix.matvec(v) for v in batch])
        assert np.allclose(ref, got, atol=TOL)

    def test_to_phase_objects_round_trip(self):
        rng = random.Random(6)
        phases = [[rng.uniform(0, 2 * math.pi) for _ in range(4)]
                  for _ in range(3)]
        matrix = PhaseMatrix(phases)
        back = matrix.to_phase_objects()
        for i in range(3):
            for j in range(4):
                assert back[i][j].phase == pytest.approx(matrix.phases[i][j])


# --------------------------------------------------------------------- #
# Attention
# --------------------------------------------------------------------- #

class TestAttentionFastEquivalence:
    def test_tpn_softmax_matches(self):
        rng = random.Random(10)
        for _ in range(50):
            rows, cols = rng.randint(1, 6), rng.randint(1, 10)
            scores = [[rng.uniform(-5, 5) for _ in range(cols)]
                      for _ in range(rows)]
            assert np.allclose(tpn_softmax(scores),
                               tpn_softmax_fast(scores), atol=TOL)

    def test_tpn_softmax_degenerate_row_matches(self):
        """All-equal rows are the divide-by-zero case; must agree."""
        scores = [[2.0, 2.0, 2.0, 2.0], [1.0, 2.0, 3.0, 4.0]]
        assert np.allclose(tpn_softmax(scores),
                           tpn_softmax_fast(scores), atol=TOL)

    def test_standard_softmax_matches(self):
        rng = random.Random(11)
        for _ in range(50):
            rows, cols = rng.randint(1, 6), rng.randint(1, 10)
            scores = [[rng.uniform(-20, 20) for _ in range(cols)]
                      for _ in range(rows)]
            assert np.allclose(standard_softmax(scores),
                               standard_softmax_fast(scores), atol=TOL)

    def test_forward_matches(self):
        rng = random.Random(12)
        for _ in range(30):
            n, m, d = rng.randint(1, 8), rng.randint(1, 8), rng.randint(1, 12)
            q = [[rng.uniform(-3, 3) for _ in range(d)] for _ in range(n)]
            k = [[rng.uniform(-3, 3) for _ in range(d)] for _ in range(m)]
            v = [[rng.uniform(-3, 3) for _ in range(d)] for _ in range(m)]
            assert np.allclose(TPNAttention.forward(q, k, v),
                               TPNAttentionFast.forward(q, k, v), atol=TOL)

    def test_forward_standard_matches(self):
        rng = random.Random(13)
        for _ in range(30):
            n, m, d = rng.randint(1, 8), rng.randint(1, 8), rng.randint(1, 12)
            q = [[rng.uniform(-3, 3) for _ in range(d)] for _ in range(n)]
            k = [[rng.uniform(-3, 3) for _ in range(d)] for _ in range(m)]
            v = [[rng.uniform(-3, 3) for _ in range(d)] for _ in range(m)]
            assert np.allclose(TPNAttention.forward_standard(q, k, v),
                               TPNAttentionFast.forward_standard(q, k, v),
                               atol=TOL)

    def test_compute_scores_matches(self):
        rng = random.Random(14)
        q = [[rng.uniform(-2, 2) for _ in range(5)] for _ in range(3)]
        k = [[rng.uniform(-2, 2) for _ in range(5)] for _ in range(4)]
        assert np.allclose(TPNAttention.compute_scores(q, k),
                           TPNAttentionFast.compute_scores(q, k), atol=TOL)

    def test_softmax_rows_sum_to_one(self):
        rng = random.Random(15)
        scores = [[rng.uniform(-5, 5) for _ in range(7)] for _ in range(4)]
        weights = tpn_softmax_fast(scores)
        assert np.allclose(weights.sum(axis=1), 1.0, atol=TOL)


# --------------------------------------------------------------------- #
# Hadamard
# --------------------------------------------------------------------- #

class TestHadamardFastEquivalence:
    @pytest.mark.parametrize("block_size", [2, 4, 8, 16, 32, 64, 128])
    def test_transform_matches_matrix_implementation(self, block_size):
        rng = random.Random(20 + block_size)
        ref = HadamardTransform(block_size=block_size)
        fast = FastHadamardTransform(block_size=block_size)

        for _ in range(10):
            length = rng.randint(1, block_size * 2)
            v = [rng.uniform(-5, 5) for _ in range(length)]
            assert np.allclose(ref.transform(v), fast.transform(v), atol=1e-9)

    def test_preserves_norm(self):
        ht = FastHadamardTransform(block_size=32)
        v = [random.uniform(-3, 3) for _ in range(32)]
        ref_norm = math.sqrt(sum(x * x for x in v))
        got_norm = math.sqrt(float(np.sum(ht.transform(v) ** 2)))
        assert got_norm == pytest.approx(ref_norm, rel=1e-10)

    def test_inverse_recovers_input(self):
        ht = FastHadamardTransform(block_size=16)
        v = [random.uniform(-3, 3) for _ in range(16)]
        assert np.allclose(ht.inverse_transform(ht.transform(v)), v, atol=1e-10)

    def test_fwht_is_involutory_up_to_n(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        assert np.allclose(fwht(fwht(x)), 4.0 * x, atol=TOL)

    def test_rejects_non_power_of_two(self):
        with pytest.raises(ValueError):
            fwht(np.ones(6))
        with pytest.raises(ValueError):
            FastHadamardTransform(block_size=6)

    def test_padding_and_truncation(self):
        ht = FastHadamardTransform(block_size=8)
        assert len(ht.transform([1.0, 2.0, 3.0])) == 8
        assert len(ht.transform([float(i) for i in range(20)])) == 8

    def test_transform_batch_matches_rows(self):
        rng = np.random.default_rng(21)
        ht = FastHadamardTransform(block_size=16)
        m = rng.normal(size=(5, 16))
        assert np.allclose(ht.transform_batch(m),
                           np.array([ht.transform(r) for r in m]), atol=1e-10)

    def test_next_power_of_two(self):
        assert next_power_of_two(1) == 1
        assert next_power_of_two(3) == 4
        assert next_power_of_two(4) == 4
        assert next_power_of_two(1000) == 1024
        assert is_power_of_two(1024) and not is_power_of_two(1000)


# --------------------------------------------------------------------- #
# Inference backends
# --------------------------------------------------------------------- #

class TestInferenceBackendEquivalence:
    """
    The numpy and python backends must be independently deterministic and
    must compute the same math.

    Because the numpy path is vectorized it uses its own independent RNG
    stream (seeded from the model seed), so it is NOT bit-identical to the
    python backend at initialization. What IS required:

    1. Same seed  -> same weights + same output  (deterministic)
    2. Different seeds -> different weights + different output
    3. Both backends initialize to the correct distribution:
       phases centered on pi/2 with spread 0.1, all in [0, 2*pi)
    4. Given identical weights, both backends produce identical forward
       outputs -- this is the real fast-path correctness guarantee.
    """

    @pytest.mark.parametrize("hidden,layers", [(16, 1), (32, 2), (64, 2)])
    def test_forward_matches(self, hidden, layers):
        """
        Same weights must give identical outputs on both backends.

        We load identical raw phase weights into EVERY matrix of each backend
        (encode=False so the phases are stored verbatim) and forward a common
        input. This isolates the computation paths (matvec -> attention ->
        feed-forward) from the init RNG differences.
        """
        config = TPNConfig(hidden_size=hidden, num_layers=layers,
                           num_heads=2, head_dim=hidden // 2,
                           intermediate_size=hidden * 2)

        # one shared random matrix per weight name, seeded deterministically
        rng_w = np.random.default_rng(0)
        intermediate = config.intermediate_size
        wmat = {name: PhaseMatrix.from_weights(
                    rng_w.uniform(-1, 1, (rows, cols)))
                for name, (rows, cols) in {
                    "q_proj": (hidden, hidden),
                    "k_proj": (hidden, hidden),
                    "v_proj": (hidden, hidden),
                    "o_proj": (hidden, hidden),
                    "gate_proj": (intermediate, hidden),
                    "up_proj": (intermediate, hidden),
                    "down_proj": (hidden, intermediate),
                }.items()}

        numpy_model = TPNModel(config, backend="numpy", seed=42)
        python_model = TPNModel(config, backend="python", seed=42)

        for li in range(layers):
            for name, phases in wmat.items():
                numpy_model.set_weight_matrix(li, name, phases.values(),
                                            encode=False)
                python_model.set_weight_matrix(li, name, phases.values().tolist(),
                                              encode=False)

        x = [0.1 * ((i % 5) - 2) for i in range(hidden)]
        out_n = numpy_model.forward(x)
        out_p = python_model.forward(x)
        assert np.allclose(out_n, out_p, atol=1e-9), \
            f"forward diverged at hidden={hidden} layers={layers}"

    def test_initialization_is_deterministic_and_valid(self):
        """Seeded init is reproducible and statistically correct."""
        config = TPNConfig(hidden_size=16, num_layers=1, num_heads=2,
                           head_dim=8, intermediate_size=32)

        # 1. Same seed -> identical weights (determinism, per backend)
        a = TPNModel(config, backend="numpy", seed=7)
        b = TPNModel(config, backend="numpy", seed=7)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"):
            assert np.allclose(a.layers[0][name].values(),
                               b.layers[0][name].values(), atol=TOL)

        c = TPNModel(config, backend="python", seed=7)
        d = TPNModel(config, backend="python", seed=7)
        for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"):
            va = [[w.value for w in row] for row in c.layers[0][name]]
            vb = [[w.value for w in row] for row in d.layers[0][name]]
            assert np.allclose(va, vb, atol=TOL), f"{name} python"

        # 2. Different seeds -> different weights and outputs
        e = TPNModel(config, backend="numpy", seed=13)
        x = [0.1] * 16
        assert not np.allclose(a.forward(x), e.forward(x), atol=1e-9)

        # 3. Both backends reach the right phase distribution
        for m, label in ((a, "numpy"), (c, "python")):
            for name in ("q_proj", "gate_proj"):
                pm = m.layers[0][name]
                ph = pm.phases if hasattr(pm, "phases") else \
                     np.array([[w.phase for w in row] for row in pm])
                assert ph.min() >= 0.0 and ph.max() < 2 * math.pi, label
                mean = float(np.mean(np.cos(ph)))
                # phase ~ N(pi/2, 0.1)  ->  cos(phase) ~ N(0, 0.1) roughly
                assert abs(mean) < 0.05, label

    def test_same_seed_is_deterministic(self):
        config = TPNConfig(hidden_size=16, num_layers=1, num_heads=2,
                           head_dim=8, intermediate_size=32)
        a = TPNModel(config, backend="numpy", seed=99)
        b = TPNModel(config, backend="numpy", seed=99)
        x = [0.1] * 16
        assert np.allclose(a.forward(x), b.forward(x), atol=TOL)

    def test_forward_batch_matches_single_forward(self):
        config = TPNConfig(hidden_size=32, num_layers=2, num_heads=2,
                           head_dim=16, intermediate_size=64)
        model = TPNModel(config, backend="numpy", seed=5)

        inputs = [[0.1 * (i + j) for j in range(32)] for i in range(4)]
        batched = model.forward_batch(inputs)
        singles = [model.forward(x) for x in inputs]
        assert np.allclose(batched, singles, atol=1e-9)

    def test_forward_batch_empty(self):
        config = TPNConfig(hidden_size=16, num_layers=1, num_heads=2,
                           head_dim=8, intermediate_size=32)
        model = TPNModel(config, backend="numpy", seed=1)
        assert model.forward_batch([]) == []

    def test_set_weight_matrix_round_trip(self):
        """Loading float weights then reading them back is lossless."""
        config = TPNConfig(hidden_size=8, num_layers=1, num_heads=2,
                           head_dim=4, intermediate_size=16)
        model = TPNModel(config, backend="numpy", seed=3)

        rng = np.random.default_rng(0)
        weights = rng.uniform(-1, 1, size=(8, 8))
        model.set_weight_matrix(0, "q_proj", weights)
        assert np.allclose(model.layers[0]["q_proj"].values(), weights,
                           atol=TOL)

    def test_set_weight_matrix_shape_mismatch_raises(self):
        config = TPNConfig(hidden_size=8, num_layers=1, num_heads=2,
                           head_dim=4, intermediate_size=16)
        model = TPNModel(config, backend="numpy", seed=3)
        with pytest.raises(ValueError):
            model.set_weight_matrix(0, "q_proj", np.zeros((3, 3)))

    def test_set_weight_matrix_unknown_name_raises(self):
        config = TPNConfig(hidden_size=8, num_layers=1, num_heads=2,
                           head_dim=4, intermediate_size=16)
        model = TPNModel(config, backend="numpy", seed=3)
        with pytest.raises(ValueError):
            model.set_weight_matrix(0, "not_a_proj", np.zeros((8, 8)))

    def test_load_block_reports(self):
        config = TPNConfig(hidden_size=8, num_layers=1, num_heads=2,
                           head_dim=4, intermediate_size=16)
        model = TPNModel(config, backend="numpy", seed=3)
        report = model.load_block(0, {
            "q_proj": np.zeros((8, 8)),
            "k_proj": np.zeros((3, 3)),   # wrong shape -> skipped
        })
        assert report["q_proj"] == "loaded"
        assert report["k_proj"].startswith("skipped")

    def test_num_parameters_matches_config(self):
        config = TPNConfig(hidden_size=64, num_layers=2, num_heads=4,
                           head_dim=16, intermediate_size=256)
        model = TPNModel(config, backend="numpy", seed=1)
        expected = (4 * 64 * 64 + 3 * 256 * 64) * 2
        assert model.num_parameters() == expected

    def test_backend_auto_selects_numpy(self):
        assert numpy_available()
        config = TPNConfig(hidden_size=8, num_layers=1, num_heads=2,
                           head_dim=4, intermediate_size=16)
        assert TPNModel(config, backend="auto", seed=1).backend == "numpy"

    def test_unknown_backend_raises(self):
        config = TPNConfig(hidden_size=8, num_layers=1, num_heads=2,
                           head_dim=4, intermediate_size=16)
        with pytest.raises(ValueError):
            TPNModel(config, backend="turbo", seed=1)

    def test_config_dict_round_trip(self):
        config = TPNConfig(hidden_size=32, num_layers=3, num_heads=4,
                           head_dim=8, intermediate_size=128)
        restored = TPNConfig.from_dict(config.to_dict())
        assert restored.to_dict() == config.to_dict()

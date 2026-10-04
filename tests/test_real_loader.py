"""
Tests for the real-checkpoint loader (sharded safetensors, bf16, mapping).

These tests use synthetic checkpoints written to a temp dir rather than a
real model, so they run anywhere and do not need 15 GB on disk. The bf16
encoding and the sharded index are exercised exactly as a real
checkpoint would use them.
"""

import json
import os
import struct
import tempfile

import numpy as np
import pytest

from tpn_engine.real_loader import (
    ShardedSafetensors,
    load_checkpoint,
    map_block_weights,
    read_hf_config,
    tpn_config_from_hf,
    _classify,
    _decode_bf16,
)


# --------------------------------------------------------------------- #
# Fixture builders
# --------------------------------------------------------------------- #

def _bf16_bytes(array: np.ndarray) -> bytes:
    """Encode float32 as bfloat16 bytes (truncate the low 16 bits)."""
    u32 = np.ascontiguousarray(array, dtype=np.float32).view(np.uint32)
    return (u32 >> 16).astype("<u2").tobytes()


def _write_shard(path, tensors):
    """Write one safetensors shard. tensors: {name: (np.array, dtype_tag)}."""
    header, blobs, offset = {}, [], 0
    for name, (arr, tag) in tensors.items():
        arr = np.ascontiguousarray(arr, dtype=np.float32)
        raw = _bf16_bytes(arr) if tag == "BF16" else arr.astype("<f4").tobytes()
        header[name] = {
            "dtype": tag,
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        blobs.append(raw)
        offset += len(raw)

    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    header_bytes += b" " * ((8 - len(header_bytes) % 8) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(header_bytes)))
        f.write(header_bytes)
        for b in blobs:
            f.write(b)


def _make_checkpoint(root, *, hidden=32, layers=2, intermediate=64,
                     vocab=40, moe=False, out_of_range=False,
                     bf16=True, flat_config=False, shards=1):
    """Build a small but structurally realistic checkpoint on disk."""
    tag = "BF16" if bf16 else "F32"
    tensors: dict = {}

    def put(name, arr):
        tensors[name] = (arr, tag)

    def scale():
        # Push past +/-1 so the out-of-range case is genuinely exercised.
        return 1.35 if out_of_range else 0.5

    for i in range(layers):
        p = f"model.layers.{i}."
        put(p + "attention.q_proj.weight",
            np.random.uniform(-scale(), scale(), (hidden, hidden)).astype(np.float32))
        put(p + "attention.k_proj.weight",
            np.random.uniform(-scale(), scale(), (hidden, hidden)).astype(np.float32))
        put(p + "attention.v_proj.weight",
            np.random.uniform(-scale(), scale(), (hidden, hidden)).astype(np.float32))
        put(p + "attention.o_proj.weight",
            np.random.uniform(-scale(), scale(), (hidden, hidden)).astype(np.float32))
        if moe:
            # Routed experts must NOT be mapped onto the dense block.
            for e in range(2):
                put(f"{p}mlp.experts.{e}.gate_proj.weight",
                    np.random.uniform(-1, 1, (intermediate, hidden)).astype(np.float32))
            put(p + "mlp.shared_experts.gate_proj.weight",
                np.random.uniform(-scale(), scale(), (intermediate, hidden)).astype(np.float32))
            put(p + "mlp.shared_experts.up_proj.weight",
                np.random.uniform(-scale(), scale(), (intermediate, hidden)).astype(np.float32))
            put(p + "mlp.shared_experts.down_proj.weight",
                np.random.uniform(-scale(), scale(), (hidden, intermediate)).astype(np.float32))
            put(p + "mlp.gate.weight", np.random.randn(2).astype(np.float32))
        else:
            put(p + "mlp.gate_proj.weight",
                np.random.uniform(-scale(), scale(), (intermediate, hidden)).astype(np.float32))
            put(p + "mlp.up_proj.weight",
                np.random.uniform(-scale(), scale(), (intermediate, hidden)).astype(np.float32))
            put(p + "mlp.down_proj.weight",
                np.random.uniform(-scale(), scale(), (hidden, intermediate)).astype(np.float32))

    put("model.word_embeddings.weight",
        np.random.uniform(-0.1, 0.1, (vocab, hidden)).astype(np.float32))
    put("model.norm.weight", np.ones(hidden, dtype=np.float32))
    put("lm_head.weight",
        np.random.uniform(-0.1, 0.1, (vocab, hidden)).astype(np.float32))

    # Distribute tensors round-robin so every shard is populated.
    all_items = list(tensors.items())
    buckets: dict = {}
    for idx, (name, payload) in enumerate(all_items):
        shard = f"model-{idx % shards:05d}-of-{shards:05d}.safetensors"
        buckets.setdefault(shard, {})[name] = payload

    for shard_name, tensors in buckets.items():
        _write_shard(os.path.join(root, shard_name), tensors)

    with open(os.path.join(root, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": {"total_size": 0},
             "weight_map": {n: f"model-{i % shards:05d}-of-{shards:05d}.safetensors"
                            for i, (n, _) in enumerate(all_items)}},
            f,
        )

    cfg = {
        "architectures": ["TestForCausalLM"],
        "model_type": "test",
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": 4,
        "head_dim": hidden // 4,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "max_position_embeddings": 512,
    }
    if moe:
        cfg["num_experts"] = 2
        cfg["num_experts_per_tok"] = 1
    if flat_config:
        cfg["hidden_size"] = hidden
        cfg["num_hidden_layers"] = layers
    with open(os.path.join(root, "config.json"), "w") as f:
        json.dump(cfg, f)

    return root


# --------------------------------------------------------------------- #
# bf16 decoding
# --------------------------------------------------------------------- #

class TestBF16:
    def test_decode_matches_reference(self):
        vals = np.array([1.0, -2.5, 0.0, 3.14159, -0.001, 1e-3],
                        dtype=np.float32)
        decoded = _decode_bf16(_bf16_bytes(vals))
        # bf16 has 8 mantissa bits, so compare at bf16 resolution.
        assert np.allclose(decoded, vals, rtol=1e-2, atol=1e-7)

    def test_decode_special_values(self):
        vals = np.array([0.0, -0.0, np.inf, -np.inf], dtype=np.float32)
        decoded = _decode_bf16(_bf16_bytes(vals))
        assert decoded[0] == 0.0 and decoded[1] == 0.0
        assert np.isposinf(decoded[2]) and np.isneginf(decoded[3])

    def test_nan_survives(self):
        decoded = _decode_bf16(_bf16_bytes(np.array([np.nan], dtype=np.float32)))
        assert np.isnan(decoded[0])


# --------------------------------------------------------------------- #
# Store
# --------------------------------------------------------------------- #

class TestShardedStore:
    def test_reads_index_and_headers(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=16, layers=2)
            s = ShardedSafetensors(d)
            assert s.num_shards() == 1
            assert len(s.tensor_names()) > 0
            assert s.layer_indices() == [0, 1]

    def test_multi_shard(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=16, layers=2, shards=4)
            s = ShardedSafetensors(d)
            assert s.num_shards() == 4
            # Every tensor must still be findable regardless of shard.
            for name in s.tensor_names():
                assert s.load(name) is not None

    def test_dtype_histogram(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=16, layers=1, bf16=True)
            s = ShardedSafetensors(d)
            assert set(s.dtypes()) == {"BF16"}

    def test_load_returns_correct_shape(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=32, layers=1)
            s = ShardedSafetensors(d)
            w = s.load("model.layers.0.attention.q_proj.weight")
            assert w.shape == (32, 32)
            assert w.dtype == np.float32

    def test_missing_tensor_returns_none(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=16, layers=1)
            s = ShardedSafetensors(d)
            assert s.load("does.not.exist") is None
            assert s.spec("does.not.exist") is None
            assert not s.has("does.not.exist")

    def test_missing_dir_raises(self):
        with pytest.raises(NotADirectoryError):
            ShardedSafetensors("/definitely/not/here")

    def test_empty_dir_raises(self):
        with tempfile.TemporaryDirectory() as d:
            with pytest.raises(FileNotFoundError):
                ShardedSafetensors(d)

    def test_stats(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=16, layers=1, out_of_range=True)
            s = ShardedSafetensors(d)
            st = s.stats("model.layers.0.attention.q_proj.weight")
            assert st["absmax"] > 1.0
            assert st["out_of_unit"] > 0


# --------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------- #

class TestConfig:
    def test_reads_flat_config(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=48, layers=3, flat_config=True)
            hf = read_hf_config(d)
            assert hf["hidden_size"] == 48
            assert hf["num_hidden_layers"] == 3

    def test_reads_nested_text_config(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d)
            cfg_path = os.path.join(d, "config.json")
            cfg = json.load(open(cfg_path))
            # Re-wrap the same values under text_config, as HF multimodal
            # and some hub checkpoints do.
            nested = {"text_config": cfg}
            json.dump(nested, open(cfg_path, "w"))
            hf = read_hf_config(d)
            assert hf["hidden_size"] == 32
            assert hf["num_hidden_layers"] == 2

    def test_tpn_config_derivation(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=32, layers=3, intermediate=64, vocab=40)
            cfg, warns = tpn_config_from_hf(read_hf_config(d))
            assert cfg.hidden_size == 32
            assert cfg.num_layers == 3
            assert cfg.intermediate_size == 64
            assert cfg.vocab_size == 40
            assert warns == []

    def test_moe_is_reported_not_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=16, layers=1, moe=True)
            cfg, warns = tpn_config_from_hf(read_hf_config(d))
            assert any("MoE" in w for w in warns)

    def test_missing_sizes_raises(self):
        with pytest.raises(ValueError):
            tpn_config_from_hf({"model_type": "x"})


# --------------------------------------------------------------------- #
# Name mapping
# --------------------------------------------------------------------- #

class TestNameMapping:
    def test_classify_plain_names(self):
        idx, key = _classify("model.layers.3.attention.q_proj.weight")
        assert idx == 3 and key == "q_proj"

    def test_classify_root_tensor(self):
        idx, key = _classify("lm_head.weight")
        assert idx is None and key == "root"

    def test_classify_strips_mlp_prefix(self):
        idx, key = _classify("model.layers.0.mlp.gate_proj.weight")
        assert idx == 0 and key == "gate_proj"

    def test_classify_shared_experts(self):
        idx, key = _classify("model.layers.5.mlp.shared_experts.down_proj.weight")
        assert idx == 5 and key == "down_proj"

    def test_map_block_weights_finds_all_seven(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=16, layers=2)
            s = ShardedSafetensors(d)
            found = map_block_weights(s, 0)
            assert set(found) == {"q_proj", "k_proj", "v_proj", "o_proj",
                                  "gate_proj", "up_proj", "down_proj"}

    def test_moe_routed_experts_not_mapped(self):
        """Per-expert weights must not be mistaken for the dense block FFN."""
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=16, layers=1, moe=True)
            s = ShardedSafetensors(d)
            found = map_block_weights(s, 0)
            for name in found.values():
                assert "experts.0." not in name
                assert "experts.1." not in name
            # The shared expert should have supplied the FFN instead.
            assert "shared_experts" in found["gate_proj"]


# --------------------------------------------------------------------- #
# End-to-end load
# --------------------------------------------------------------------- #

class TestLoadCheckpoint:
    def test_loads_weights_into_model(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=32, layers=2, intermediate=64)
            model, report = load_checkpoint(d)
            assert model is not None
            assert len(report.loaded) == 2 * 7
            assert not report.skipped

    def test_weights_are_phase_encoded_and_used(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=32, layers=1)
            model, _ = load_checkpoint(d)
            s = ShardedSafetensors(d)
            original = s.load("model.layers.0.attention.q_proj.weight")
            stored = model.layers[0]["q_proj"].values()
            assert np.allclose(stored, original, atol=1e-6)

    def test_out_of_range_weights_are_not_clamped(self):
        """Values beyond +/-1 must survive the phase encoding."""
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=32, layers=1, out_of_range=True)
            model, _ = load_checkpoint(d)
            s = ShardedSafetensors(d)
            original = s.load("model.layers.0.attention.q_proj.weight")
            assert np.abs(original).max() > 1.0, "fixture must exercise this"
            stored = model.layers[0]["q_proj"].values()
            assert np.allclose(stored, original, atol=1e-6)
            # The clamp would have flattened every |w| > 1 to exactly 1.
            assert np.abs(stored).max() > 1.0

    def test_max_layers_limits_work(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=16, layers=3)
            model, report = load_checkpoint(d, max_layers=1)
            assert len(report.loaded) == 7
            assert any("max_layers" in w for w in report.warnings)

    def test_inspect_only_returns_no_model(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=16, layers=1)
            model, report = load_checkpoint(d, create_model=False)
            assert model is None
            assert any("no model built" in w for w in report.warnings)

    def test_forward_uses_loaded_weights(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=32, layers=1, intermediate=64)
            model, _ = load_checkpoint(d)
            out = model.forward([0.1] * 32)
            assert len(out) == 32
            assert all(np.isfinite(out))

    def test_different_weights_give_different_output(self):
        """A loaded model must actually depend on the loaded values."""
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            _make_checkpoint(d1, hidden=32, layers=1)
            _make_checkpoint(d2, hidden=32, layers=1)
            m1, _ = load_checkpoint(d1)
            m2, _ = load_checkpoint(d2)
            x = [0.1] * 32
            assert not np.allclose(m1.forward(x), m2.forward(x))

    def test_multi_shard_load(self):
        with tempfile.TemporaryDirectory() as d:
            _make_checkpoint(d, hidden=16, layers=2, shards=3)
            model, report = load_checkpoint(d)
            assert model is not None
            assert len(report.loaded) == 14

    def test_bf16_and_f32_agree(self):
        """The bf16 path must decode to (nearly) the f32 values it stored."""
        with tempfile.TemporaryDirectory() as d1, tempfile.TemporaryDirectory() as d2:
            np.random.seed(1234)
            _make_checkpoint(d1, hidden=32, layers=1, bf16=True)
            np.random.seed(1234)
            _make_checkpoint(d2, hidden=32, layers=1, bf16=False)
            s1 = ShardedSafetensors(d1)
            s2 = ShardedSafetensors(d2)
            a = s1.load("model.layers.0.attention.q_proj.weight")
            b = s2.load("model.layers.0.attention.q_proj.weight")
            assert set(s1.dtypes()) == {"BF16"}
            assert set(s2.dtypes()) == {"F32"}
            assert np.allclose(a, b, rtol=1e-2, atol=1e-3)

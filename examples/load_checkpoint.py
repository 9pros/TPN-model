"""
End-to-end: load a real (dense) checkpoint into a TPNModel and run inference.

Usage:
    python examples/load_checkpoint.py /path/to/checkpoint

The example also builds a tiny dense checkpoint from scratch if no directory
is given, so it can be run as-is without downloading anything.

The Ling MoE-MLA checkpoints are NOT dense: their layers use routed experts
and LoRA-projected attention (q_lora_rank / kv_lora_rank), so their tensor
shapes do not fit TPN's dense SwiGLU block. The loader detects that and
rejects them loudly instead of silently producing a broken model. Use a
dense checkpoint (or a dense TPN-exported one) with this script.
"""

import os
import sys
import tempfile
import time

import numpy as np

from tpn_engine.real_loader import load_checkpoint, ShardedSafetensors

# --------------------------------------------------------------------- #
# Fixture builder: one small dense safetensors checkpoint on disk
# --------------------------------------------------------------------- #

def _write_checkpoint(path, hidden=32, layers=2, intermediate=64, vocab=40):
    """Write a minimal but structurally dense checkpoint in safetensors."""
    import struct, json

    tensors: dict = {}
    def put(key, arr):
        tensors[key] = np.ascontiguousarray(arr, dtype=np.float32)
    total = 0
    for i in range(layers):
        for suf in ("k_proj", "v_proj", "o_proj"):
            key = f"model.layers.{i}.attention.{suf}.weight"
            shape = (hidden, hidden) if suf != "o_proj" else (hidden, hidden)
            tensors[key] = np.random.default_rng(i + 10).normal(0, 0.5, shape).astype(np.float32)
        for suf in ("gate_proj", "up_proj"):
            tensors[f"model.layers.{i}.mlp.{suf}.weight"] = np.random.default_rng(i + 20)\
                .normal(0, 0.5, (intermediate, hidden)).astype(np.float32)
        tensors[f"model.layers.{i}.mlp.down_proj.weight"] = np.random.default_rng(i + 30)\
            .normal(0, 0.5, (hidden, intermediate)).astype(np.float32)
    tensors["model.word_embeddings.weight"] = np.random.default_rng(100)\
        .normal(0, 0.1, (vocab, hidden)).astype(np.float32)
    tensors["model.norm.weight"] = np.ones(hidden, dtype=np.float32)
    tensors["lm_head.weight"] = np.random.default_rng(101)\
        .normal(0, 0.1, (vocab, hidden)).astype(np.float32)


    total = sum(t.nbytes for t in tensors.values())
    shard_name = path if path.endswith(".safetensors") else f"{path}.safetensors"
    header = {n: {"dtype": "F32", "shape": list(v.shape),
                  "data_offsets": [0, 0]} for n, v in tensors.items()}
    off = 0
    blobs = []
    for name, v in tensors.items():
        raw = v.tobytes()
        header[name]["data_offsets"] = [off, off + len(raw)]
        blobs.append(raw)
        off += len(raw)
    hdr = json.dumps(header, separators=(",", ":")).encode("utf-8")
    hdr += b" " * ((8 - len(hdr) % 8) % 8)
    with open(shard_name, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        for b in blobs:
            f.write(b)
    with open(f"{shard_name}.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total},
                   "weight_map": {n: os.path.basename(shard_name)
                                  for n in tensors}}, f)

    json.dump({
        "architectures": ["TPNForCausalLM"],
        "hidden_size": hidden,
        "num_hidden_layers": layers,
        "num_attention_heads": 4,
        "head_dim": hidden // 4,
        "intermediate_size": intermediate,
        "vocab_size": vocab,
        "max_position_embeddings": 512,
    }, open(f"{path}.json", "w"))


# --------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------- #

def main():
    if len(sys.argv) > 1:
        model_dir = sys.argv[1]
        built = False
    else:
        model_dir = tempfile.mkdtemp(prefix="tpn_demo_")
        fname = f"{model_dir}/model.safetensors"
        _write_checkpoint(fname)
        # config.json next to the shard, like HF does.
        import json as _j
        _j.dump({"architectures": ["TPNForCausalLM"],
                 "hidden_size": 32, "num_hidden_layers": 2,
                 "num_attention_heads": 4, "head_dim": 8,
                 "intermediate_size": 64, "vocab_size": 40,
                 "max_position_embeddings": 512},
                open(f"{model_dir}/config.json", "w"))
        print(f"built demo dense checkpoint in {model_dir}")
        built = True

    print(f"loading {model_dir} ...")
    t0 = time.time()
    model, report = load_checkpoint(model_dir)
    dt = time.time() - t0
    print(f"loaded in {dt:.2f}s")
    print("  loaded:", len(report.loaded), "tensors")
    if report.warnings:
        print("  warnings:", "; ".join(report.warnings))
    if report.skipped:
        print("  skipped:", "; ".join(report.skipped.keys()))

    if model is None:
        print("\nCould not build a model from this checkpoint (it is likely a")
        print("MoE/MLA or otherwise non-dense architecture). The loader reported")
        print("the reasons above; that is the correct, safe outcome.")
        return

    # Round-trip check: stored weights must equal the checkpoint values.
    s = ShardedSafetensors(model_dir)
    for name in ("q_proj", "k_proj", "v_proj", "o_proj",
                 "gate_proj", "up_proj", "down_proj"):
        ref = s.load(f"model.layers.0.attention.{name}.weight")
        if ref is None:
            ref = s.load(f"model.layers.0.mlp.{name}.weight")
        if ref is None:
            continue
        stored = np.array([r for r in model.layers[0][name].values()])
        err = np.abs(stored - ref).max()
        print(f"  round-trip {name}: max abs error = {err:.2e}")

    # Run inference and show the weights are actually doing something.
    x1 = [0.1] * 32
    t0 = time.time()
    out1 = model.forward(x1)
    dt1 = time.time() - t0
    print(f"forward(32): {len(out1)} tokens, {dt1*1000:.2f} ms, "
          f"finite={all(np.isfinite(out1))}")
    # Varied inputs produce varied outputs: weights are wired into the graph.
    # Constant-valued inputs can collapse through normalization, so use a
    # rising sequence for this check.
    x2 = list(range(32))
    out2 = model.forward(x2)
    print("weights wired in:", not np.allclose(out1, out2))

    if built:
        print()
        print("(demo checkpoint is at", model_dir, ")")


if __name__ == "__main__":
    main()

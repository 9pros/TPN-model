import os, sys, tempfile
import json, struct
import numpy as np

# Quick dense checkpoint fixture
hidden, layers, intermediate, vocab = 32, 2, 64, 40
d = tempfile.mkdtemp(prefix="wired_")
tensors = {}
rng = np.random.default_rng(0)
for i in range(layers):
    for suf in ("q_proj", "k_proj", "v_proj", "o_proj"):
        tensors[f"model.layers.{i}.attention.{suf}.weight"] = rng.normal(0, 0.5, (hidden, hidden)).astype(np.float32)
    for suf in ("gate_proj", "up_proj"):
        tensors[f"model.layers.{i}.mlp.{suf}.weight"] = rng.normal(0, 0.5, (intermediate, hidden)).astype(np.float32)
    tensors[f"model.layers.{i}.mlp.down_proj.weight"] = rng.normal(0, 0.5, (hidden, intermediate)).astype(np.float32)
tensors["model.word_embeddings.weight"] = rng.normal(0, 0.1, (vocab, hidden)).astype(np.float32)
tensors["model.norm.weight"] = np.ones(hidden)
tensors["lm_head.weight"] = rng.normal(0, 0.1, (vocab, hidden)).astype(np.float32)

hdr, blobs, off = {}, [], 0
for n, v in tensors.items():
    raw = v.tobytes()
    hdr[n] = {"dtype": "F32", "shape": list(v.shape), "data_offsets": [off, off + len(raw)]}
    blobs.append(raw); off += len(raw)
hdr = json.dumps(hdr, separators=(",", ":")).encode("utf-8") + b" " * ((8 - len(hdr) % 8) % 8)
p = f"{d}/model.safetensors"
with open(p, "wb") as f:
    f.write(struct.pack("<Q", len(hdr))); f.write(hdr); f.writelines(blobs)
json.dump({"weight_map": {n: "model.safetensors" for n in tensors}},
          open(f"{d}/model.safetensors.index.json", "w"))
json.dump({"architectures": ["TPNForCausalLM"], "hidden_size": hidden,
           "num_hidden_layers": layers, "num_attention_heads": 4, "head_dim": 8,
           "intermediate_size": intermediate, "vocab_size": vocab,
           "max_position_embeddings": 512},
          open(f"{d}/config.json", "w"))

from tpn_engine.real_loader import load_checkpoint

model, report = load_checkpoint(d)
print("loaded:", len(report.loaded), "skipped:", len(report.skipped))

x1 = np.arange(32, dtype=np.float32) / 32.0
x2 = np.arange(32, dtype=np.float32)[::-1]
o1 = model.forward(x1)
o2 = model.forward(x2)
print("varied inputs produce different outputs:", not np.allclose(o1, o2))
print("all finite:", all(np.isfinite(o1)))
print("output1[:4]:", list(o1[:4]))
print("output2[:4]:", list(o2[:4]))

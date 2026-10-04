#!/usr/bin/env python3
"""
End-to-end generation with the TPNModel.

Builds a tiny dense checkpoint on disk (vocab=32, hidden=16, 2 layers),
loads it with ``load_checkpoint`` (proving weights reach the model),
and runs ``model.generate`` with the Ling tokenizer and the temporal
context for stateful, blended generation.

Run:
    python3 examples/generate.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tpn_engine import TPNConfig, TPNModel
from tpn_engine.inference import TPNModel as Model
from tpn_engine.real_loader import load_checkpoint, LoadReport
from tpn_engine.tokenizer import Tokenizer
from tpn_engine.temporal_context import TemporalContext


CFG = TPNConfig(hidden_size=16, num_layers=2, intermediate_size=32,
                vocab_size=32, max_position_embeddings=512)


def _write_dense_fixture(path: str, checkpoint_dir: str) -> None:
    """Build a minimal dense checkpoint (bf16-ish float32) on disk."""
    rng = __import__('random').Random(7)

    def rand_matrix(rows: int, cols: int) -> list:
        return [[rng.uniform(-1.0, 1.0) for _ in range(cols)]
                for _ in range(rows)]

    tensors = {}
    for layer in range(CFG.num_layers):
        tensors[f"model.layers.{layer}.q_proj"] = rand_matrix(CFG.hidden_size,
                                                              CFG.hidden_size)
        tensors[f"model.layers.{layer}.k_proj"] = rand_matrix(CFG.hidden_size,
                                                              CFG.hidden_size)
        tensors[f"model.layers.{layer}.v_proj"] = rand_matrix(CFG.hidden_size,
                                                              CFG.hidden_size)
        tensors[f"model.layers.{layer}.o_proj"] = rand_matrix(CFG.hidden_size,
                                                              CFG.hidden_size)
        tensors[f"model.layers.{layer}.gate_proj"] = rand_matrix(CFG.intermediate_size,
                                                                 CFG.hidden_size)
        tensors[f"model.layers.{layer}.up_proj"] = rand_matrix(CFG.intermediate_size,
                                                               CFG.hidden_size)
        tensors[f"model.layers.{layer}.down_proj"] = rand_matrix(CFG.hidden_size,
                                                                 CFG.intermediate_size)
    tensors["word_embeddings.weight"] = rand_matrix(CFG.vocab_size,
                                                    CFG.hidden_size)
    tensors["lm_head.weight"] = rand_matrix(CFG.vocab_size, CFG.hidden_size)
    # Point index keys at the shard basename (the loader matches by basename).
    index = {"metadata": {"total_size": 0},
             "weight_map": {name: "dense.safetensors" for name in tensors}}
    with open(os.path.join(path, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f)

    # Copy the real checkpoint's tokenizer so generation tokens the vocabulary.
    ling_tok = os.path.join(checkpoint_dir, "tokenizer.json")
    if os.path.isfile(ling_tok):
        import shutil
        shutil.copy(ling_tok, os.path.join(path, "tokenizer.json"))

    try:
        import numpy as np
        import safetensors.numpy as sn
    except ImportError:
        raise SystemExit(
            "safetensors + numpy needed for the on-disk fixture. "
            "pip install safetensors numpy")
    arr = {name: np.array(m, dtype=np.float32) for name, m in tensors.items()}
    sn.save_file(arr, os.path.join(path, "dense.safetensors"))

    config = {
        "architectures": ["TPNModel"],
        "hidden_size": CFG.hidden_size,
        "num_hidden_layers": CFG.num_layers,
        "intermediate_size": CFG.intermediate_size,
        "num_attention_heads": CFG.hidden_size // 8,
        "vocab_size": CFG.vocab_size,
        "max_position_embeddings": CFG.max_position_embeddings,
    }
    with open(os.path.join(path, "config.json"), "w") as f:
        json.dump(config, f)


def main() -> int:
    fixture_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                               "fixtures", "dense_fixture")
    os.makedirs(fixture_dir, exist_ok=True)
    print(f"Building dense fixture in {fixture_dir} ...")
    _write_dense_fixture(fixture_dir, "/Users/binetic/models/ling-3.0-tiny-bf16")

    print("Loading checkpoint into TPNModel ...")
    model, report = load_checkpoint(fixture_dir, max_layers=CFG.num_layers)
    print(f"  report: {len(report.loaded)} loaded, "
          f"{len(report.skipped)} skipped")
    if report.skipped:
        print("  skipped:", dict(list(report.skipped.items())[:4]))

    print("Loading tokenizer ...")
    tok = Tokenizer.from_checkpoint(fixture_dir)
    print(f"  vocab: {len(tok)}")

    print("Running model.generate(prompt='the model is ', max_tokens=8) ...")
    context = TemporalContext(dim=CFG.hidden_size, num_frequencies=8,
                              window=16.0, forgetting=0.995, seed=1)
    generated = model.generate(
        "the model is ",
        tok,
        max_tokens=8,
        context=context,
        blend_memory=True,
        temperature=0.9,
        top_k=8,
        seed=42,
        verbose=True,
        return_tokens=True,
    )
    text, tokens = generated
    print(f"\nGenerated ({len(tokens)} tokens):")
    print(f"  {text}")
    print(f"  token ids: {tokens}")

    print("\nSaving temporal context to disk ...")
    ctx_path = os.path.join(fixture_dir, "context.json")
    model.save_context(ctx_path, context)
    print(f"  saved {ctx_path}")

    print("Loading context back and continuing generation ...")
    resumed = model.load_context(ctx_path)
    cont = model.generate(" continuing now", tok, max_tokens=6,
                          context=resumed, blend_memory=True,
                          temperature=0.9, top_k=8, seed=42, verbose=True)
    print(f"\nContinuation: {cont}")

    print(f"\nTotal parameters (incl. word_embeddings + lm_head): "
          f"{model.num_parameters()}")

    # Determinism check: generate twice with the same seed -> identical output.
    ctx2 = TemporalContext(dim=CFG.hidden_size, num_frequencies=8,
                           window=16.0, forgetting=0.995, seed=1)
    g1 = model.generate("the model is ", tok, max_tokens=4, context=ctx2,
                        seed=999, return_tokens=True)
    ctx3 = TemporalContext(dim=CFG.hidden_size, num_frequencies=8,
                           window=16.0, forgetting=0.995, seed=1)
    g2 = model.generate("the model is ", tok, max_tokens=4, context=ctx3,
                        seed=999, return_tokens=True)
    print("Determinism check (same seed, same context init):",
          g1 == g2)

    return 0


if __name__ == "__main__":
    sys.exit(main())

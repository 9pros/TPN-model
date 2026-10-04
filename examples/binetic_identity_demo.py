#!/usr/bin/env python3
"""
Binetic identity demo: prove the model knows it's Binetic.

Every authored, trained, or converted model carries a provable Binetic
identity (owner binetic.ai / partners Max Yeremenko & binetic-partner)
in the .binetic file header AND on the loaded model.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
from tpn_engine import TPNConfig, TPNModel
from tpn_engine.binetic import BineticFormat


def main() -> int:
    out = "/tmp/binetic_identity_demo.binetic"

    # --- 1. Fresh model: NOT Binetic yet ---
    print("1. Fresh TPNModel -> is_binetic:", TPNModel(TPNConfig()).is_binetic())
    assert not TPNModel(TPNConfig()).is_binetic()

    # --- 2. Train the model: evolve() auto-registers generations ---
    print("2. Training (evolve)...")
    config = TPNConfig(hidden_size=64, num_layers=2, intermediate_size=256,
                       vocab_size=512)
    model = TPNModel(config)
    target = [math.sin(i * 0.1) for i in range(64)]
    result = model.evolve(target, generations=20, mutation_rate=0.2)
    print(f"   final fitness: {result['final_fitness']:.4f}")

    # --- 3. Convert -> the converter ALWAYS writes Binetic ownership ---
    print("3. Saving to .binetic (identity injected by converter)...")
    model.save_to_binetic(out, version_id="train_v1")

    # --- 4. Loaded model PROVES it is Binetic ---
    print("4. Loaded model:")
    loaded = TPNModel(config)
    loaded.load_from_binetic(out)
    print(f"   is_binetic: {loaded.is_binetic()}")
    print(f"   owner: {loaded.binetic_identity.owner}")
    print(f"   tool: {loaded.binetic_identity.tool}")
    print(f"   partners: {loaded.binetic_identity.partners}")
    print(f"   coauthors: {loaded.binetic_identity.coauthors}")
    hist = loaded.binetic_identity.generation_info.get("evolution_history", [])
    print(f"   evolution runs: {len(hist)}")
    print(f"   best fitness: {hist[-1]['final_fitness'] if hist else None}")
    assert loaded.is_binetic()
    assert loaded.binetic_identity.owner == "binetic.ai"
    assert loaded.binetic_identity.partners == ["Max Yeremenko", "binetic-partner"]

    # --- 5. The file header PROVES authorship standalone ---
    print("5. File header identity (verifiable offline):")
    fmt = BineticFormat()
    hdr = fmt.identity_of(out)
    print(f"   header owner: {hdr.owner}")
    print(f"   header tool: {hdr.tool}")
    print(f"   header generated_at: {hdr.created_at}")
    assert hdr.owner == "binetic.ai"

    # --- 6. Version history lives per BineticFormat instance; the
    # saved v1 graph + auto-saved demo_roll graph are trackable in one
    # instance.
    fmt.auto_save(out, loaded.binetic_graph, inference_id="demo_roll")
    hist = fmt.get_version_history(out)
    print(f"6. Version history entries (this instance): {len(hist)}")
    for h in hist:
        print(f"   - {h.version_id} @ {h.timestamp:.0f}")
    assert len(hist) >= 1

    print()
    print("CONCLUSION: the model knows it is Binetic, and every .binetic file")
    print("proves Binetic ownership by Max Yeremenko & binetic-partner.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

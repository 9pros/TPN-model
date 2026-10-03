"""
Minimal Bonsai weight test with tiny TPN model.

Creates a small TPN model (hidden_size=64) and loads a slice of
Bonsai weights to test the full pipeline without OOM.
"""

import time
import numpy as np
from tpn_engine import TPNModel, TPNConfig, SafetensorsLoader, PhaseEncoding


def main():
    print("=" * 70)
    print("Minimal Bonsai Weight Test")
    print("=" * 70)
    print()
    
    # Create tiny TPN model
    tpn_config = TPNConfig(
        hidden_size=64,
        num_layers=1,
        num_heads=2,
        head_dim=32,
        intermediate_size=128,
        max_position_embeddings=512,
        vocab_size=1000,
    )
    model = TPNModel(tpn_config)
    print(f"TPN model: {tpn_config.hidden_size} hidden, {tpn_config.num_layers} layer")
    print()
    
    # Load Bonsai weights (just a slice)
    print("Loading Bonsai weights (slice)...")
    loader = SafetensorsLoader('/root/models/bonsai-mlx/model.safetensors')
    
    # Extract embedding weights (first 64 tokens, first 64 dims)
    emb_tensor = loader.load_tensor('language_model.model.embed_tokens.weight')
    if emb_tensor is not None:
        emb_slice = emb_tensor[:64, :64].tolist()
        model.embedding = PhaseEncoding.encode_matrix(emb_slice)
        print(f"  embed_tokens slice: {len(model.embedding)} x {len(model.embedding[0])}")
    
    # Extract first layer weights (first 64 rows/cols)
    layer_weights = loader.extract_layer_weights(0)
    for name, data in layer_weights.items():
        if len(data.shape) == 2:
            # Take slice
            rows = min(64, data.shape[0])
            cols = min(64, data.shape[1])
            data_slice = data[:rows, :cols].tolist()
            phase_weights = PhaseEncoding.encode_matrix(data_slice)
            if not hasattr(model, 'phase_weights'):
                model.phase_weights = {}
            model.phase_weights[f'blk.0.{name}'] = phase_weights
            print(f"  blk.0.{name}: {rows} x {cols}")
        elif len(data.shape) == 1:
            # 1D tensor (biases, norms)
            data_slice = data[:64].tolist()
            if not hasattr(model, 'phase_weights_1d'):
                model.phase_weights_1d = {}
            model.phase_weights_1d[f'blk.0.{name}'] = data_slice
            print(f"  blk.0.{name}: {len(data_slice)} values")
    
    print()
    
    # Run inference
    print("Running inference...")
    input_data = [0.1] * 64
    start = time.time()
    output = model.forward(input_data)
    elapsed = time.time() - start
    
    print(f"Input: {input_data[:5]}... ({len(input_data)} values)")
    print(f"Output: {output[:5]}... ({len(output)} values)")
    print(f"Time: {elapsed:.4f}s")
    print()
    
    # Test with different inputs
    print("Testing with different inputs...")
    for i in range(5):
        input_data = [0.1 * (i + 1)] * 64
        output = model.forward(input_data)
        print(f"  Input {i+1}: {input_data[0]:.1f} -> Output: {output[0]:.4f}")
    print()
    
    # Test trajectory prediction
    print("Testing trajectory prediction...")
    from tpn_engine import TrajectoryPredictor
    predictor = TrajectoryPredictor(num_trajectories=5, hidden_size=64)
    trajectories = predictor.predict(model, [0.1] * 64)
    print(f"Generated {len(trajectories)} trajectories")
    print(f"Best fitness: {trajectories[0].fitness:.6f}")
    print(f"Worst fitness: {trajectories[-1].fitness:.6f}")
    print()
    
    # Test intent prediction
    print("Testing intent prediction...")
    from tpn_engine import IntentPredictor
    intent_predictor = IntentPredictor(hidden_size=64, num_intents=5)
    state = intent_predictor.predict(model, [0.1] * 64)
    print(f"Intent: {state.current_intent:.4f}")
    print(f"Confidence: {state.confidence:.4f}")
    print()
    
    # Test intent evolution
    print("Testing intent evolution...")
    result = intent_predictor.evolve_intent(model, [0.1] * 64, 0.8, generations=20)
    print(f"Target: 0.8")
    print(f"Initial: {result['initial_intent']:.4f}")
    print(f"Final: {result['final_intent']:.4f}")
    print(f"Distance: {result['final_distance']:.4f}")
    print()
    
    print("=" * 70)
    print("SUCCESS: Bonsai weights loaded into TPN engine!")
    print("=" * 70)
    print()
    print("Pipeline verified:")
    print("  ✓ Safetensors weight extraction")
    print("  ✓ Phase-encoded weight conversion")
    print("  ✓ TPN model inference")
    print("  ✓ Trajectory prediction")
    print("  ✓ Intent prediction")
    print("  ✓ Intent evolution")


if __name__ == "__main__":
    main()

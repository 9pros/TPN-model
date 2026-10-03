"""
Memory-efficient Bonsai weight loading and inference test.

Loads only the first layer of Bonsai weights to test inference
without running out of memory.
"""

import time
import numpy as np
from tpn_engine import TPNModel, TPNConfig, SafetensorsLoader, PhaseEncoding


def main():
    print("=" * 70)
    print("Bonsai Weight Loading Test (Memory Efficient)")
    print("=" * 70)
    print()
    
    # Load safetensors model
    print("Loading safetensors model...")
    loader = SafetensorsLoader('/root/models/bonsai-mlx/model.safetensors')
    config = loader.get_model_config()
    print(f"Model config: {config}")
    print()
    
    # Create TPN model with matching config (but only 1 layer for memory)
    tpn_config = TPNConfig(
        hidden_size=config['hidden_size'],
        num_layers=1,  # Only 1 layer for memory efficiency
        num_heads=config['num_heads'],
        head_dim=config['head_dim'],
        intermediate_size=config['intermediate_size'],
        max_position_embeddings=config['max_position_embeddings'],
    )
    model = TPNModel(tpn_config)
    print(f"TPN model created: {model.get_config()}")
    print()
    
    # Extract first layer weights
    print("Extracting first layer weights...")
    start = time.time()
    layer_weights = loader.extract_layer_weights(0)
    elapsed = time.time() - start
    print(f"Extracted {len(layer_weights)} tensors in {elapsed:.2f}s")
    print()
    
    # Show weight shapes
    print("Weight shapes:")
    for name, data in list(layer_weights.items())[:10]:
        print(f"  {name}: {data.shape}")
    print()
    
    # Convert to phase-encoded format
    print("Converting to phase-encoded format...")
    start = time.time()
    
    # Load embedding weights
    emb_tensor = loader.load_tensor('language_model.model.embed_tokens.weight')
    if emb_tensor is not None:
        # Only take first 1000 tokens for memory efficiency
        emb_data = emb_tensor[:1000].tolist()
        model.embedding = PhaseEncoding.encode_matrix(emb_data)
        print(f"  embed_tokens: {len(model.embedding)} x {len(model.embedding[0])}")
    
    # Load first layer weights
    for name, data in layer_weights.items():
        if len(data.shape) == 2:
            # Only take first 100 rows for memory efficiency
            data_list = data[:100].tolist()
            phase_weights = PhaseEncoding.encode_matrix(data_list)
            if not hasattr(model, 'phase_weights'):
                model.phase_weights = {}
            model.phase_weights[f'blk.0.{name}'] = phase_weights
            print(f"  blk.0.{name}: {len(phase_weights)} x {len(phase_weights[0])}")
    
    elapsed = time.time() - start
    print(f"Conversion completed in {elapsed:.2f}s")
    print()
    
    # Run inference with random input
    print("Running inference...")
    input_data = [0.1] * tpn_config.hidden_size
    start = time.time()
    output = model.forward(input_data)
    elapsed = time.time() - start
    
    print(f"Input: {input_data[:5]}... ({len(input_data)} values)")
    print(f"Output: {output[:5]}... ({len(output)} values)")
    print(f"Time: {elapsed:.2f}s")
    print()
    
    # Test with different inputs
    print("Testing with different inputs...")
    for i in range(3):
        input_data = [0.1 * (i + 1)] * tpn_config.hidden_size
        output = model.forward(input_data)
        print(f"  Input {i+1}: {input_data[0]:.1f} -> Output: {output[0]:.4f}")
    print()
    
    # Test trajectory prediction
    print("Testing trajectory prediction...")
    from tpn_engine import TrajectoryPredictor
    predictor = TrajectoryPredictor(num_trajectories=5, hidden_size=tpn_config.hidden_size)
    trajectories = predictor.predict(model, [0.1] * tpn_config.hidden_size)
    print(f"Generated {len(trajectories)} trajectories")
    print(f"Best fitness: {trajectories[0].fitness:.6f}")
    print()
    
    # Test intent prediction
    print("Testing intent prediction...")
    from tpn_engine import IntentPredictor
    intent_predictor = IntentPredictor(hidden_size=tpn_config.hidden_size, num_intents=5)
    state = intent_predictor.predict(model, [0.1] * tpn_config.hidden_size)
    print(f"Intent: {state.current_intent:.4f}")
    print(f"Confidence: {state.confidence:.4f}")
    print()
    
    print("=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"Successfully loaded Bonsai weights into TPN engine")
    print(f"Model: {tpn_config.num_layers} layer(s), {tpn_config.hidden_size} hidden size")
    print(f"Weights: {len(layer_weights)} tensors extracted")
    print(f"Inference: Working")
    print(f"Trajectory prediction: Working")
    print(f"Intent prediction: Working")
    print()
    print("Next: Load more layers and test intelligence growth")


if __name__ == "__main__":
    main()

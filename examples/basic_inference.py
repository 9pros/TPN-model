"""
Basic TPN inference example.
"""

from tpn_engine import TPNModel, TPNConfig


def main():
    print("=" * 60)
    print("TPN Inference Engine - Basic Example")
    print("=" * 60)
    print()
    
    # Create model
    config = TPNConfig(
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        intermediate_size=256,
    )
    model = TPNModel(config)
    
    print(f"Model config: {model.get_config()}")
    print()
    
    # Run inference
    input_data = [0.1] * 128
    print(f"Input: {input_data[:5]}... (128 values)")
    
    output = model.forward(input_data)
    print(f"Output: {output[:5]}... (128 values)")
    print()
    
    # Evolve
    target = [0.5] * 128
    print(f"Evolving toward target: {target[:5]}... (128 values)")
    
    result = model.evolve(target, generations=50)
    print(f"Evolution result: {result}")
    print()
    
    print("Done!")


if __name__ == "__main__":
    main()

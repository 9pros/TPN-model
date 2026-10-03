"""
TPN Evolution Demo.
"""

import math
from tpn_engine import TPNModel, TPNConfig


def main():
    print("=" * 60)
    print("TPN Evolution Demo")
    print("=" * 60)
    print()
    
    # Create model
    config = TPNConfig(
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        head_dim=32,
    )
    model = TPNModel(config)
    
    # Create target pattern
    target = [math.sin(i * 0.1) for i in range(64)]
    print(f"Target: {target[:5]}... (64 values)")
    print()
    
    # Evolve
    print("Evolving...")
    result = model.evolve(target, generations=100, mutation_rate=0.2)
    
    print(f"Final fitness: {result['final_fitness']:.6f}")
    print(f"Generations: {result['generations']}")
    print()
    
    # Test evolved model
    input_data = [0.0] * 64
    output = model.forward(input_data)
    print(f"Output: {output[:5]}... (64 values)")
    print()
    
    print("Done!")


if __name__ == "__main__":
    main()

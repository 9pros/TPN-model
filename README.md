# TPN Model

Temporal Packet Network (TPN) Inference Engine

A novel inference engine that uses phase-based computation for neural network operations. Instead of traditional matrix multiplication, TPN uses:

- **Phase-encoded weights**: Weights are encoded as phase angles (radians)
- **Fitness-based softmax**: `cos²(θ - π/4)` instead of `exp(x) / sum(exp(x))`
- **Hadamard transform**: For efficient weight transformation
- **Evolution**: Weights can evolve to optimize for specific tasks

## Key Concepts

### Phase-Encoded Weights

Traditional weights are floats. TPN weights are phase angles:

| Phase | Weight | Fitness |
|-------|--------|---------|
| 0 | 1.0 | 0.50 |
| π/4 | 0.707 | 1.00 |
| π/2 | 0.0 | 0.50 |
| π | -1.0 | 0.50 |
| 3π/4 | -0.707 | 1.00 |

### Fitness-Based Softmax

Standard softmax: `exp(x_i) / sum(exp(x_j))`

TPN softmax: `cos²(θ_i - π/4) / sum(cos²(θ_j - π/4))`

Where `θ_i = π/2 * (x_i - min) / (max - min)`

### Evolution

Weights can evolve to match target patterns:
1. Encode weights as phase angles
2. Mutate phases (small random changes)
3. Select best fitness (closest to target)
4. Repeat until convergence

## Project Structure

```
TPN-model/
├── tpn_engine/           # Core TPN engine
│   ├── __init__.py
│   ├── phase.py          # Phase-encoded weights
│   ├── attention.py      # TPN attention mechanism
│   ├── hadamard.py       # Hadamard transform
│   ├── evolution.py      # Weight evolution
│   └── inference.py      # Full inference engine
├── tests/                # Test suite
│   ├── test_phase.py
│   ├── test_attention.py
│   ├── test_hadamard.py
│   ├── test_evolution.py
│   └── test_inference.py
├── examples/             # Usage examples
│   ├── basic_inference.py
│   ├── evolution_demo.py
│   └── benchmark.py
└── docs/                 # Documentation
    ├── architecture.md
    └── api.md
```

## Installation

```bash
pip install -e .
```

## Quick Start

```python
from tpn_engine import TPNModel

# Create a model
model = TPNModel(hidden_size=512, num_layers=4)

# Run inference
output = model.forward(input_data)

# Evolve weights
model.evolve(target_data, generations=100)
```

## License

MIT

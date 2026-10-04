# TPN Model

Temporal Packet Network (TPN) Inference Engine

A novel inference engine that uses phase-based computation for neural network
operations. Instead of traditional matrix multiplication, TPN uses:

- **Phase-encoded weights**: Weights are encoded as phase angles (radians)
- **Fitness-based softmax**: `cos²(θ - π/4)` instead of `exp(x) / sum(exp(x))`
- **Hadamard transform**: For efficient weight transformation
- **Evolution**: Weights can evolve to optimize for specific tasks
- **Fast numpy backend**: ~40–1,450x faster than the reference pure-Python path

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

## Performance

All optimizations preserve the original math exactly — every fast path is
verified elementwise against the reference implementation in `tests/`.

| Operation | Speedup |
|-----------|---------|
| Weight → matrix matvec | ~40–49x |
| Attention (TPN softmax) | ~8–150x |
| Hadamard transform | ~19–1,440x |
| 4-layer, 512-hidden forward pass | ~41–50x (2.27 s → 53 ms) |
| Memory (512-hidden, block=1024) | ~7x (940 MB → 134 MB) |

The numpy path is used automatically when numpy is installed (it is now a
core dependency). The pure-Python path remains for reference and for
environments without numpy.

## Installation

```bash
pip install -e .
```

### Extras

```bash
# Test suite
pip install -e ".[test]"

# GGUF loader support
pip install -e ".[gguf]"
```

## Usage

### Basic inference

```python
from tpn_engine import TPNModel

model = TPNModel(hidden_size=512, num_layers=4)
output = model.forward(input_data)
model.evolve(target_data, generations=100)
```

### Loading a dense checkpoint

```python
from tpn_engine.real_loader import load_checkpoint

model, report = load_checkpoint("/path/to/dense/checkpoint")
print(report.summary())
output = model.forward(input_data)
```

The loader opens sharded safetensors checkpoints on disk, maps block tensors
to TPN layers, and verifies the round-trip. Every unmappable tensor is
reported in the `report` rather than silently dropped.

**MoE / MLA checkpoints are not supported.** Models with routed experts
(`num_experts`) or LoRA-projected/MLA attention (`q_lora_rank`, `kv_lora_rank`)
are detected and rejected loudly with the reason. TPN is a dense SwiGLU
architecture, so those checkpoints must first be converted to a dense form.

## Project Structure

```
TPN-model/
├── tpn_engine/           # Core TPN engine
│   ├── __init__.py
│   ├── phase.py          # Phase-encoded weights (PhaseMatrix)
│   ├── attention.py      # TPN attention mechanism
│   ├── hadamard.py       # Hadamard transform
│   ├── evolution.py      # Weight evolution
│   ├── inference.py      # Full inference engine (numpy + python backends)
│   ├── attention_fast.py # Vectorized attention (numpy path)
│   ├── hadamard_fast.py  # O(N log N) FWHT
│   ├── real_loader.py    # Sharded safetensors checkpoint loader
│   └── ...               # trajectory, intent, temporal_context, echo,
│                          # tool_calling, bitslicing, binetic, etc.
├── tests/                # Test suite (230+ tests)
│   ├── test_phase.py
│   ├── test_attention.py
│   ├── test_hadamard.py
│   ├── test_evolution.py
│   ├── test_inference.py
│   ├── test_fast_paths.py       # Fast-path vs reference equivalence
│   ├── test_real_loader.py      # Loader correctness on fixtures
│   └── ...
├── examples/             # Usage examples
│   ├── basic_inference.py
│   ├── evolution_demo.py
│   ├── benchmark.py
│   ├── load_checkpoint.py     # Dense checkpoint → forward pass
│   └── ...
└── docs/                 # Documentation
    ├── architecture.md
    └── api.md
```

## License

MIT

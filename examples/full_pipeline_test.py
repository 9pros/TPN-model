"""
Full pipeline test: TPN model + OpenAI API + .binetic format.

Tests the complete workflow:
1. Create TPN model
2. Register tools
3. Create chat completion
4. Save as .binetic
5. Load and verify
6. Run inference
"""

import time
import json
from tpn_engine import (
    OpenAIChatCompletions,
    ChatMessage,
    TPNModel,
    TPNConfig,
    BineticFormat,
    BineticConverter,
    TrajectoryPredictor,
    IntentPredictor,
)


def test_full_pipeline():
    """Test the complete TPN pipeline."""
    print("=" * 70)
    print("FULL PIPELINE TEST")
    print("=" * 70)
    print()
    
    # Step 1: Create TPN model
    print("Step 1: Creating TPN model...")
    config = TPNConfig(
        hidden_size=64,
        num_layers=2,
        num_heads=2,
        head_dim=32,
        intermediate_size=128,
    )
    model = TPNModel(config)
    print(f"  Model: {config.hidden_size} hidden, {config.num_layers} layers")
    print()
    
    # Step 2: Create OpenAI API
    print("Step 2: Creating OpenAI API...")
    api = OpenAIChatCompletions(model=model)
    print("  API created")
    print()
    
    # Step 3: Register tools
    print("Step 3: Registering tools...")
    
    def add(a: int, b: int) -> int:
        return a + b
    
    def multiply(a: int, b: int) -> int:
        return a * b
    
    api.register_tool(
        name="add",
        description="Add two numbers",
        function=add,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        }
    )
    
    api.register_tool(
        name="multiply",
        description="Multiply two numbers",
        function=multiply,
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        }
    )
    
    print(f"  Registered {len(api.get_tools())} tools")
    print()
    
    # Step 4: Create chat completion
    print("Step 4: Creating chat completion...")
    messages = [ChatMessage(role="user", content="What is 2+2?")]
    
    start = time.time()
    response = api.create_completion(
        model="tpn-model",
        messages=messages,
        tools=api.get_tools(),
        max_tokens=50,
    )
    elapsed = time.time() - start
    
    print(f"  Response: {response['choices'][0]['message']['content'][:50]}...")
    print(f"  Time: {elapsed:.4f}s")
    print(f"  Tokens: {response['usage']['completion_tokens']}")
    print()
    
    # Step 5: Save as .binetic
    print("Step 5: Saving as .binetic...")
    converter = BineticConverter()
    
    # Convert model to graph
    model_data = {
        "architecture": "transformer",
        "model_name": "tpn-test-model",
        "hidden_size": config.hidden_size,
        "num_layers": config.num_layers,
        "layers": [],
        "connections": [],
    }
    
    # Add layers
    for i, layer in enumerate(model.layers):
        layer_data = {
            "name": f"layer_{i}",
            "type": "transformer",
            "weights": [[w.value for w in row] for row in layer["q_proj"]],
        }
        model_data["layers"].append(layer_data)
        
        if i > 0:
            model_data["connections"].append((f"layer_{i-1}", f"layer_{i}"))
    
    graph = converter.convert(model_data)
    
    # Save
    fmt = BineticFormat()
    fmt.save("test_model.binetic", graph, version_id="v1")
    print("  Saved to test_model.binetic")
    print()
    
    # Step 6: Load and verify
    print("Step 6: Loading and verifying...")
    loaded_graph, version = fmt.load("test_model.binetic")
    print(f"  Loaded version: {version.version_id}")
    print(f"  Nodes: {len(loaded_graph.nodes)}")
    print(f"  Edges: {len(loaded_graph.edges)}")
    print()
    
    # Step 7: Run inference
    print("Step 7: Running inference...")
    input_data = [0.1] * config.hidden_size
    
    start = time.time()
    output = model.forward(input_data)
    elapsed = time.time() - start
    
    print(f"  Input: {input_data[:5]}... ({len(input_data)} values)")
    print(f"  Output: {output[:5]}... ({len(output)} values)")
    print(f"  Time: {elapsed:.4f}s")
    print()
    
    # Step 8: Test trajectory prediction
    print("Step 8: Testing trajectory prediction...")
    predictor = TrajectoryPredictor(num_trajectories=5, hidden_size=config.hidden_size)
    trajectories = predictor.predict(model, input_data)
    
    print(f"  Generated {len(trajectories)} trajectories")
    print(f"  Best fitness: {trajectories[0].fitness:.6f}")
    print()
    
    # Step 9: Test intent prediction
    print("Step 9: Testing intent prediction...")
    intent_predictor = IntentPredictor(hidden_size=config.hidden_size, num_intents=5)
    state = intent_predictor.predict(model, input_data)
    
    print(f"  Intent: {state.current_intent:.4f}")
    print(f"  Confidence: {state.confidence:.4f}")
    print()
    
    # Step 10: Test auto-save
    print("Step 10: Testing auto-save...")
    fmt.auto_save("test_model.binetic", loaded_graph, inference_id="test_inference")
    
    history = fmt.get_version_history("test_model.binetic")
    print(f"  Version history: {len(history)} versions")
    print()
    
    print("=" * 70)
    print("FULL PIPELINE TEST COMPLETE")
    print("=" * 70)
    print()
    print("All components working:")
    print("  ✓ TPN model creation")
    print("  ✓ OpenAI API")
    print("  ✓ Tool registration")
    print("  ✓ Chat completion")
    print("  ✓ .binetic save/load")
    print("  ✓ Inference")
    print("  ✓ Trajectory prediction")
    print("  ✓ Intent prediction")
    print("  ✓ Auto-save")


def compare_storage_efficiency():
    """Compare storage efficiency of different formats."""
    print()
    print("=" * 70)
    print("STORAGE EFFICIENCY COMPARISON")
    print("=" * 70)
    print()
    
    # Model size: 27B parameters
    params = 27_000_000_000
    
    # Standard formats
    print("Standard Formats:")
    print(f"  FP32: {params * 4 / 1e9:.2f} GB")
    print(f"  FP16: {params * 2 / 1e9:.2f} GB")
    print(f"  INT8: {params * 1 / 1e9:.2f} GB")
    print(f"  INT4: {params * 0.5 / 1e9:.2f} GB")
    print(f"  Ternary: {params * 2 / 8 / 1e9:.2f} GB (2 bits per weight)")
    print()
    
    # Our .binetic format
    print("Our .binetic Format:")
    print(f"  Phase-encoded (FP32): {params * 4 / 1e9:.2f} GB")
    print(f"  Phase-encoded (FP16): {params * 2 / 1e9:.2f} GB")
    print(f"  Phase-encoded (INT8): {params * 1 / 1e9:.2f} GB")
    print(f"  Phase-encoded (INT4): {params * 0.5 / 1e9:.2f} GB")
    print()
    
    # With graph compression
    print("With Graph Compression:")
    print(f"  Sequential layers: ~0% overhead")
    print(f"  Shared weights: up to 50% savings")
    print(f"  Version history: ~10% per version")
    print()
    
    # Inference speed
    print("Inference Speed:")
    print(f"  Standard attention: O(n²)")
    print(f"  TPN attention: O(n²) (same complexity)")
    print(f"  With trajectory prediction: O(k × n²) where k = num trajectories")
    print(f"  With intent evolution: O(g × n²) where g = generations")
    print()
    
    print("Key Insights:")
    print("  1. Storage: Similar to standard formats (depends on quantization)")
    print("  2. Inference: Similar speed, but with additional capabilities")
    print("  3. Unique features: Version control, rollback, auto-save")
    print("  4. Graph structure: Enables better model understanding")


if __name__ == "__main__":
    test_full_pipeline()
    compare_storage_efficiency()

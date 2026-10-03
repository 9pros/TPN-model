"""
Tests for universal model to .binetic converter.
"""

import pytest
from tpn_engine.binetic_converter import BineticConverter


class TestBineticConverter:
    """Test universal model to .binetic conversion."""
    
    def test_converter_creation(self):
        """Test creating a binetic converter."""
        converter = BineticConverter()
        assert converter is not None
    
    def test_convert_simple_model(self):
        """Test converting a simple model."""
        converter = BineticConverter()
        
        # Simple model: 2 layers
        model_data = {
            "layers": [
                {"name": "layer_0", "weights": [[0.1, 0.2], [0.3, 0.4]]},
                {"name": "layer_1", "weights": [[0.5, 0.6], [0.7, 0.8]]},
            ],
            "connections": [("layer_0", "layer_1")],
        }
        
        graph = converter.convert(model_data)
        
        assert "layer_0" in graph.nodes
        assert "layer_1" in graph.nodes
        assert ("layer_0", "layer_1") in graph.edges
    
    def test_convert_with_metadata(self):
        """Test converting with model metadata."""
        converter = BineticConverter()
        
        model_data = {
            "model_name": "test_model",
            "hidden_size": 64,
            "num_layers": 2,
            "layers": [
                {"name": "layer_0", "weights": [[0.1, 0.2]]},
                {"name": "layer_1", "weights": [[0.3, 0.4]]},
            ],
            "connections": [("layer_0", "layer_1")],
        }
        
        graph = converter.convert(model_data)
        
        assert graph.metadata["model_name"] == "test_model"
        assert graph.metadata["hidden_size"] == 64
    
    def test_convert_transformer(self):
        """Test converting a transformer model."""
        converter = BineticConverter()
        
        # Transformer: embed → attn → ffn → output
        model_data = {
            "architecture": "transformer",
            "layers": [
                {"name": "embed", "type": "embedding", "weights": [[0.1, 0.2]]},
                {"name": "attn_q", "type": "attention", "weights": [[0.3, 0.4]]},
                {"name": "attn_k", "type": "attention", "weights": [[0.5, 0.6]]},
                {"name": "attn_v", "type": "attention", "weights": [[0.7, 0.8]]},
                {"name": "ffn", "type": "ffn", "weights": [[0.9, 1.0]]},
                {"name": "output", "type": "output", "weights": [[1.1, 1.2]]},
            ],
            "connections": [
                ("embed", "attn_q"),
                ("embed", "attn_k"),
                ("embed", "attn_v"),
                ("attn_q", "ffn"),
                ("attn_k", "ffn"),
                ("attn_v", "ffn"),
                ("ffn", "output"),
            ],
        }
        
        graph = converter.convert(model_data)
        
        assert len(graph.nodes) == 6
        assert len(graph.edges) == 7
        assert graph.metadata["architecture"] == "transformer"
    
    def test_convert_cnn(self):
        """Test converting a CNN model."""
        converter = BineticConverter()
        
        # CNN: conv → pool → conv → pool → fc
        model_data = {
            "architecture": "cnn",
            "layers": [
                {"name": "conv1", "type": "conv2d", "weights": [[0.1, 0.2]]},
                {"name": "pool1", "type": "maxpool", "weights": None},
                {"name": "conv2", "type": "conv2d", "weights": [[0.3, 0.4]]},
                {"name": "pool2", "type": "maxpool", "weights": None},
                {"name": "fc", "type": "linear", "weights": [[0.5, 0.6]]},
            ],
            "connections": [
                ("conv1", "pool1"),
                ("pool1", "conv2"),
                ("conv2", "pool2"),
                ("pool2", "fc"),
            ],
        }
        
        graph = converter.convert(model_data)
        
        assert len(graph.nodes) == 5
        assert len(graph.edges) == 4
        assert graph.metadata["architecture"] == "cnn"
    
    def test_convert_rnn(self):
        """Test converting an RNN model."""
        converter = BineticConverter()
        
        # RNN: embed → lstm → output
        model_data = {
            "architecture": "rnn",
            "layers": [
                {"name": "embed", "type": "embedding", "weights": [[0.1, 0.2]]},
                {"name": "lstm", "type": "lstm", "weights": [[0.3, 0.4]]},
                {"name": "output", "type": "output", "weights": [[0.5, 0.6]]},
            ],
            "connections": [
                ("embed", "lstm"),
                ("lstm", "output"),
            ],
        }
        
        graph = converter.convert(model_data)
        
        assert len(graph.nodes) == 3
        assert len(graph.edges) == 2
        assert graph.metadata["architecture"] == "rnn"
    
    def test_convert_quantized_model(self):
        """Test converting a quantized model."""
        converter = BineticConverter()
        
        # Quantized model: weights are integers
        model_data = {
            "architecture": "quantized",
            "quantization": {"bits": 2, "mode": "affine"},
            "layers": [
                {"name": "layer_0", "weights": [[1, 0], [0, 1]]},  # Ternary
                {"name": "layer_1", "weights": [[-1, 1], [1, -1]]},
            ],
            "connections": [("layer_0", "layer_1")],
        }
        
        graph = converter.convert(model_data)
        
        assert "layer_0" in graph.nodes
        assert graph.metadata["quantization"]["bits"] == 2
    
    def test_convert_from_gguf(self):
        """Test converting from GGUF format."""
        converter = BineticConverter()
        
        # Simulate GGUF model data
        gguf_data = {
            "architecture": "qwen2",
            "tensors": {
                "token_embd.weight": [[0.1, 0.2], [0.3, 0.4]],
                "output.weight": [[0.5, 0.6], [0.7, 0.8]],
            },
            "layers": [
                {"name": "embed", "weights": [[0.1, 0.2]]},
                {"name": "output", "weights": [[0.5, 0.6]]},
            ],
            "connections": [("embed", "output")],
        }
        
        graph = converter.convert(gguf_data)
        
        assert "embed" in graph.nodes
        assert "output" in graph.nodes
    
    def test_convert_from_safetensors(self):
        """Test converting from safetensors format."""
        converter = BineticConverter()
        
        # Simulate safetensors model data
        safetensors_data = {
            "architecture": "transformer",
            "tensors": {
                "language_model.model.embed_tokens.weight": [[0.1, 0.2]],
                "language_model.lm_head.weight": [[0.3, 0.4]],
            },
            "layers": [
                {"name": "embed", "weights": [[0.1, 0.2]]},
                {"name": "output", "weights": [[0.3, 0.4]]},
            ],
            "connections": [("embed", "output")],
        }
        
        graph = converter.convert(safetensors_data)
        
        assert "embed" in graph.nodes
        assert "output" in graph.nodes
    
    def test_save_converted_model(self):
        """Test saving converted model to .binetic."""
        import tempfile
        import os
        
        converter = BineticConverter()
        
        model_data = {
            "model_name": "test_model",
            "layers": [
                {"name": "layer_0", "weights": [[0.1, 0.2]]},
                {"name": "layer_1", "weights": [[0.3, 0.4]]},
            ],
            "connections": [("layer_0", "layer_1")],
        }
        
        graph = converter.convert(model_data)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = os.path.join(tmpdir, "converted.binetic")
            
            from tpn_engine import BineticFormat
            fmt = BineticFormat()
            fmt.save(filepath, graph, version_id="v1")
            
            assert os.path.exists(filepath)
            
            # Load and verify
            loaded_graph, _ = fmt.load(filepath)
            assert "layer_0" in loaded_graph.nodes
            assert loaded_graph.metadata["model_name"] == "test_model"

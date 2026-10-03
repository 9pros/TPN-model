"""
Tests for GGUF weight loader.
"""

import pytest
from tpn_engine.gguf_loader import GGUFLoader


class TestGGUFLoader:
    """Test GGUF weight loading."""
    
    def test_loader_creation(self):
        """Test creating a GGUF loader."""
        loader = GGUFLoader('/root/models/test/qwen2.5-0.5b-instruct-q8_0.gguf')
        assert loader.model_path == '/root/models/test/qwen2.5-0.5b-instruct-q8_0.gguf'
    
    def test_load_tensor_names(self):
        """Test loading tensor names."""
        loader = GGUFLoader('/root/models/test/qwen2.5-0.5b-instruct-q8_0.gguf')
        names = loader.get_tensor_names()
        assert len(names) > 0
        assert 'token_embd.weight' in names
        assert 'output.weight' in names
    
    def test_load_tensor(self):
        """Test loading a single tensor."""
        loader = GGUFLoader('/root/models/test/qwen2.5-0.5b-instruct-q8_0.gguf')
        tensor = loader.load_tensor('token_embd.weight')
        assert tensor is not None
        assert len(tensor.shape) == 2
    
    def test_load_tensor_data(self):
        """Test loading tensor data."""
        loader = GGUFLoader('/root/models/test/qwen2.5-0.5b-instruct-q8_0.gguf')
        tensor = loader.load_tensor('blk.0.attn_q.weight')
        assert tensor is not None
        assert tensor.shape == (896, 896)
    
    def test_get_model_config(self):
        """Test getting model configuration from GGUF."""
        loader = GGUFLoader('/root/models/test/qwen2.5-0.5b-instruct-q8_0.gguf')
        config = loader.get_model_config()
        assert config['hidden_size'] == 896
        assert config['num_layers'] == 24
        assert config['intermediate_size'] == 4864
    
    def test_extract_weights(self):
        """Test extracting all weights."""
        loader = GGUFLoader('/root/models/test/qwen2.5-0.5b-instruct-q8_0.gguf')
        weights = loader.extract_weights()
        assert 'token_embd.weight' in weights
        assert 'output.weight' in weights
        assert 'blk.0.attn_q.weight' in weights
        assert 'blk.0.attn_k.weight' in weights
        assert 'blk.0.attn_v.weight' in weights
        assert 'blk.0.attn_output.weight' in weights
        assert 'blk.0.ffn_gate.weight' in weights
        assert 'blk.0.ffn_up.weight' in weights
        assert 'blk.0.ffn_down.weight' in weights

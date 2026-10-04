"""
Tests for safetensors weight loader.
"""

import os
import pytest
from tpn_engine.safetensors_loader import SafetensorsLoader

# Path to the test fixture (a minimal safetensors file)
FIXTURE_DIR = os.path.join(os.path.dirname(__file__), 'fixtures')
FIXTURE_PATH = os.path.join(FIXTURE_DIR, 'model.safetensors')
FIXTURE_CONFIG = os.path.join(FIXTURE_DIR, 'config.json')


class TestSafetensorsLoader:
    """Test safetensors weight loading."""

    def test_loader_creation(self):
        """Test creating a safetensors loader."""
        loader = SafetensorsLoader(FIXTURE_PATH)
        assert loader.model_path == FIXTURE_PATH

    def test_load_tensor_names(self):
        """Test loading tensor names."""
        loader = SafetensorsLoader(FIXTURE_PATH)
        names = loader.get_tensor_names()
        assert len(names) > 0
        assert 'language_model.model.embed_tokens.weight' in names

    def test_load_tensor(self):
        """Test loading a single tensor."""
        loader = SafetensorsLoader(FIXTURE_PATH)
        tensor = loader.load_tensor('language_model.model.embed_tokens.weight')
        assert tensor is not None
        assert len(tensor.shape) == 2

    def test_get_model_config(self):
        """Test getting model configuration."""
        loader = SafetensorsLoader(FIXTURE_PATH)
        config = loader.get_model_config()
        assert config['hidden_size'] == 5120
        assert config['num_layers'] == 64
        assert config['num_heads'] == 24

    def test_extract_layer_weights(self):
        """Test extracting weights for a specific layer."""
        loader = SafetensorsLoader(FIXTURE_PATH)
        weights = loader.extract_layer_weights(0)
        assert len(weights) > 0
        assert 'input_layernorm.weight' in weights

    def test_extract_weights_memory_efficient(self):
        """Test extracting weights without loading all into memory."""
        loader = SafetensorsLoader(FIXTURE_PATH)
        # Only load first layer
        weights = loader.extract_layer_weights(0)
        assert len(weights) > 0
        # Check that weights are numpy arrays
        import numpy as np
        for name, data in weights.items():
            assert isinstance(data, np.ndarray)

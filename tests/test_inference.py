"""
Tests for full TPN inference engine.
"""

import math
import pytest
from tpn_engine.inference import TPNModel, TPNConfig


class TestTPNConfig:
    """Test TPN configuration."""
    
    def test_default_config(self):
        """Test default configuration."""
        config = TPNConfig()
        assert config.hidden_size > 0
        assert config.num_layers > 0
        assert config.num_heads > 0
        assert config.head_dim > 0
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = TPNConfig(hidden_size=256, num_layers=4, num_heads=4, head_dim=64)
        assert config.hidden_size == 256
        assert config.num_layers == 4
        assert config.num_heads == 4
        assert config.head_dim == 64


class TestTPNModel:
    """Test TPN inference model."""
    
    def test_model_creation(self):
        """Test creating a TPN model."""
        config = TPNConfig(hidden_size=64, num_layers=2, num_heads=2, head_dim=32)
        model = TPNModel(config)
        assert model.config.hidden_size == 64
        assert model.config.num_layers == 2
    
    def test_model_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        config = TPNConfig(hidden_size=64, num_layers=2, num_heads=2, head_dim=32)
        model = TPNModel(config)
        input_data = [0.0] * 64
        output = model.forward(input_data)
        assert len(output) == 64
    
    def test_model_forward_deterministic(self):
        """Test that forward pass is deterministic."""
        config = TPNConfig(hidden_size=64, num_layers=2, num_heads=2, head_dim=32)
        model = TPNModel(config)
        input_data = [0.0] * 64
        output1 = model.forward(input_data)
        output2 = model.forward(input_data)
        for o1, o2 in zip(output1, output2):
            assert o1 == pytest.approx(o2, rel=1e-6)
    
    def test_model_forward_different_inputs(self):
        """Test that different inputs produce different outputs."""
        config = TPNConfig(hidden_size=64, num_layers=2, num_heads=2, head_dim=32)
        model = TPNModel(config)
        input1 = [0.0] * 64
        input2 = [1.0] * 64
        output1 = model.forward(input1)
        output2 = model.forward(input2)
        # Outputs should be different
        assert output1 != output2
    
    def test_model_evolve(self):
        """Test evolving the model."""
        config = TPNConfig(hidden_size=64, num_layers=2, num_heads=2, head_dim=32)
        model = TPNModel(config)
        target = [0.0] * 64
        result = model.evolve(target, generations=10)
        assert result["generations"] == 10
        assert "final_fitness" in result
    
    def test_model_get_config(self):
        """Test getting model configuration."""
        config = TPNConfig(hidden_size=64, num_layers=2, num_heads=2, head_dim=32)
        model = TPNModel(config)
        info = model.get_config()
        assert info["hidden_size"] == 64
        assert info["num_layers"] == 2

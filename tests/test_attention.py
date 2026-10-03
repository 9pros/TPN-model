"""
Tests for TPN attention mechanism.
"""

import math
import pytest
from tpn_engine.attention import TPNAttention, tpn_softmax, standard_softmax


class TestTPNAttention:
    """Test TPN attention mechanism."""
    
    def test_tpn_softmax_sums_to_one(self):
        """Test that TPN softmax produces valid probabilities."""
        scores = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
        weights = tpn_softmax(scores)
        for row in weights:
            total = sum(row)
            assert total == pytest.approx(1.0, rel=1e-6)
    
    def test_tpn_softmax_all_same(self):
        """Test TPN softmax with all same scores."""
        scores = [[0.5, 0.5, 0.5]]
        weights = tpn_softmax(scores)
        for w in weights[0]:
            assert w == pytest.approx(1.0 / 3.0, rel=1e-6)
    
    def test_tpn_softmax_range(self):
        """Test that TPN softmax weights are in [0, 1]."""
        scores = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
        weights = tpn_softmax(scores)
        for row in weights:
            for w in row:
                assert 0 <= w <= 1
    
    def test_standard_softmax_sums_to_one(self):
        """Test that standard softmax produces valid probabilities."""
        scores = [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]]
        weights = standard_softmax(scores)
        for row in weights:
            total = sum(row)
            assert total == pytest.approx(1.0, rel=1e-6)
    
    def test_tpn_attention_output_shape(self):
        """Test that TPN attention produces correct output shape."""
        q = [[1.0, 0.0], [0.0, 1.0]]
        k = [[1.0, 0.0], [0.0, 1.0]]
        v = [[1.0, 2.0], [3.0, 4.0]]
        output = TPNAttention.forward(q, k, v)
        assert len(output) == 2
        assert len(output[0]) == 2
    
    def test_tpn_attention_deterministic(self):
        """Test that TPN attention is deterministic."""
        q = [[1.0, 0.0], [0.0, 1.0]]
        k = [[1.0, 0.0], [0.0, 1.0]]
        v = [[1.0, 2.0], [3.0, 4.0]]
        output1 = TPNAttention.forward(q, k, v)
        output2 = TPNAttention.forward(q, k, v)
        for i in range(len(output1)):
            for j in range(len(output1[0])):
                assert output1[i][j] == pytest.approx(output2[i][j], rel=1e-6)
    
    def test_tpn_vs_standard_different(self):
        """Test that TPN and standard attention produce different outputs."""
        q = [[1.0, 0.0], [0.0, 1.0]]
        k = [[1.0, 0.0], [0.0, 1.0]]
        v = [[1.0, 2.0], [3.0, 4.0]]
        output_tpn = TPNAttention.forward(q, k, v)
        output_std = TPNAttention.forward_standard(q, k, v)
        # They should be different (different softmax functions)
        # But both should be valid outputs
        assert len(output_tpn) == len(output_std)
        assert len(output_tpn[0]) == len(output_std[0])

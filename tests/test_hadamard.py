"""
Tests for Hadamard transform.
"""

import math
import pytest
from tpn_engine.hadamard import HadamardTransform


class TestHadamardTransform:
    """Test Hadamard transform functionality."""
    
    def test_hadamard_creation(self):
        """Test creating a Hadamard transform."""
        ht = HadamardTransform(block_size=8)
        assert ht.block_size == 8
    
    def test_hadamard_transform_preserves_norm(self):
        """Test that Hadamard transform preserves vector norm."""
        ht = HadamardTransform(block_size=8)
        vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        original_norm = math.sqrt(sum(x * x for x in vector))
        transformed = ht.transform(vector)
        transformed_norm = math.sqrt(sum(x * x for x in transformed))
        assert original_norm == pytest.approx(transformed_norm, rel=1e-6)
    
    def test_hadamard_inverse(self):
        """Test that inverse transform recovers original."""
        ht = HadamardTransform(block_size=8)
        vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
        transformed = ht.transform(vector)
        recovered = ht.inverse_transform(transformed)
        for orig, rec in zip(vector, recovered):
            assert orig == pytest.approx(rec, rel=1e-6)
    
    def test_hadamard_transform_pads_short_vector(self):
        """Test that short vectors are padded."""
        ht = HadamardTransform(block_size=8)
        vector = [1.0, 2.0, 3.0]
        transformed = ht.transform(vector)
        assert len(transformed) == 8
    
    def test_hadamard_transform_truncates_long_vector(self):
        """Test that long vectors are truncated."""
        ht = HadamardTransform(block_size=8)
        vector = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
        transformed = ht.transform(vector)
        assert len(transformed) == 8
    
    def test_hadamard_transform_zero_vector(self):
        """Test transform of zero vector."""
        ht = HadamardTransform(block_size=8)
        vector = [0.0] * 8
        transformed = ht.transform(vector)
        for t in transformed:
            assert t == pytest.approx(0.0, rel=1e-6)
    
    def test_hadamard_transform_ones_vector(self):
        """Test transform of ones vector."""
        ht = HadamardTransform(block_size=8)
        vector = [1.0] * 8
        transformed = ht.transform(vector)
        # First element should be sqrt(8), rest should be 0
        assert transformed[0] == pytest.approx(math.sqrt(8), rel=1e-6)
        for t in transformed[1:]:
            assert t == pytest.approx(0.0, rel=1e-6)

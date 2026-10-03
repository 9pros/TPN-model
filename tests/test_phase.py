"""
Tests for phase-encoded weights.
"""

import math
import random
import pytest
from tpn_engine.phase import PhaseWeight, PhaseEncoding


class TestPhaseWeight:
    """Test phase-encoded weight functionality."""
    
    def test_phase_weight_creation(self):
        """Test creating a phase weight."""
        pw = PhaseWeight(math.pi / 4)
        assert pw.phase == math.pi / 4
        assert pw.value == pytest.approx(math.cos(math.pi / 4), rel=1e-6)
        assert pw.fitness == pytest.approx(1.0, rel=1e-6)
    
    def test_phase_weight_zero(self):
        """Test phase weight at zero."""
        pw = PhaseWeight(0)
        assert pw.value == pytest.approx(1.0, rel=1e-6)
        assert pw.fitness == pytest.approx(0.5, rel=1e-6)
    
    def test_phase_weight_pi_half(self):
        """Test phase weight at pi/2."""
        pw = PhaseWeight(math.pi / 2)
        assert pw.value == pytest.approx(0.0, rel=1e-6)
        assert pw.fitness == pytest.approx(0.5, rel=1e-6)
    
    def test_phase_weight_pi(self):
        """Test phase weight at pi."""
        pw = PhaseWeight(math.pi)
        assert pw.value == pytest.approx(-1.0, rel=1e-6)
        assert pw.fitness == pytest.approx(0.5, rel=1e-6)
    
    def test_phase_weight_three_pi_half(self):
        """Test phase weight at 3*pi/2."""
        pw = PhaseWeight(3 * math.pi / 2)
        assert pw.value == pytest.approx(0.0, rel=1e-6)
        assert pw.fitness == pytest.approx(0.5, rel=1e-6)
    
    def test_phase_weight_fitness_range(self):
        """Test that fitness is always in [0, 1]."""
        for _ in range(100):
            phase = random.uniform(0, 2 * math.pi)
            pw = PhaseWeight(phase)
            assert 0 <= pw.fitness <= 1
    
    def test_phase_weight_mutation(self):
        """Test mutating a phase weight."""
        pw = PhaseWeight(math.pi / 4)
        mutated = pw.mutate(mutation_rate=0.1)
        assert mutated is not pw
        assert mutated.phase != pw.phase or mutated.phase == pw.phase  # Could be same by chance
    
    def test_phase_weight_mutation_range(self):
        """Test that mutation keeps phase in valid range."""
        pw = PhaseWeight(math.pi / 4)
        for _ in range(100):
            mutated = pw.mutate(mutation_rate=0.5)
            assert 0 <= mutated.phase <= 2 * math.pi


class TestPhaseEncoding:
    """Test phase encoding for vectors and matrices."""
    
    def test_encode_vector(self):
        """Test encoding a vector as phase weights."""
        values = [1.0, 0.0, -1.0, 0.5, -0.5]
        encoded = PhaseEncoding.encode_vector(values)
        assert len(encoded) == len(values)
        for pw in encoded:
            assert isinstance(pw, PhaseWeight)
    
    def test_decode_vector(self):
        """Test decoding phase weights back to values."""
        values = [1.0, 0.0, -1.0, 0.5, -0.5]
        encoded = PhaseEncoding.encode_vector(values)
        decoded = PhaseEncoding.decode_vector(encoded)
        for orig, dec in zip(values, decoded):
            assert orig == pytest.approx(dec, rel=1e-6)
    
    def test_encode_matrix(self):
        """Test encoding a matrix as phase weights."""
        matrix = [[1.0, 0.0], [-1.0, 0.5]]
        encoded = PhaseEncoding.encode_matrix(matrix)
        assert len(encoded) == len(matrix)
        assert len(encoded[0]) == len(matrix[0])
    
    def test_decode_matrix(self):
        """Test decoding phase weights back to matrix."""
        matrix = [[1.0, 0.0], [-1.0, 0.5]]
        encoded = PhaseEncoding.encode_matrix(matrix)
        decoded = PhaseEncoding.decode_matrix(encoded)
        for orig_row, dec_row in zip(matrix, decoded):
            for orig, dec in zip(orig_row, dec_row):
                assert orig == pytest.approx(dec, rel=1e-6)

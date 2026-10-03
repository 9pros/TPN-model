"""
Tests for weight evolution.
"""

import math
import pytest
from tpn_engine.evolution import EvolvablePhaseNetwork


class TestEvolvablePhaseNetwork:
    """Test evolvable phase network functionality."""
    
    def test_network_creation(self):
        """Test creating an evolvable phase network."""
        network = EvolvablePhaseNetwork(num_weights=10)
        assert len(network.weights) == 10
        assert network.generation == 0
    
    def test_compute_fitness(self):
        """Test computing fitness."""
        network = EvolvablePhaseNetwork(num_weights=5)
        target = [0.0, 0.0, 0.0, 0.0, 0.0]
        fitness = network.compute_fitness(target)
        assert 0 <= fitness <= 1
    
    def test_evolve_improves_fitness(self):
        """Test that evolution improves fitness."""
        network = EvolvablePhaseNetwork(num_weights=10)
        target = [math.sin(i * 0.1) for i in range(10)]
        
        initial_fitness = network.compute_fitness(target)
        result = network.evolve(target, mutation_rate=0.2, generations=50)
        final_fitness = result["final_fitness"]
        
        assert final_fitness >= initial_fitness
    
    def test_evolve_generations(self):
        """Test that evolution runs for specified generations."""
        network = EvolvablePhaseNetwork(num_weights=5)
        target = [0.0] * 5
        result = network.evolve(target, mutation_rate=0.1, generations=20)
        assert result["generations"] == 20
    
    def test_evolve_fitness_history(self):
        """Test that fitness history is recorded."""
        network = EvolvablePhaseNetwork(num_weights=5)
        target = [0.0] * 5
        result = network.evolve(target, mutation_rate=0.1, generations=10)
        assert len(result["fitness_history"]) == 11  # initial + 10 generations
    
    def test_evolve_deterministic_with_seed(self):
        """Test that evolution is deterministic with same seed."""
        import random
        random.seed(42)
        network1 = EvolvablePhaseNetwork(num_weights=5)
        target = [0.0] * 5
        result1 = network1.evolve(target, mutation_rate=0.1, generations=10)
        
        random.seed(42)
        network2 = EvolvablePhaseNetwork(num_weights=5)
        result2 = network2.evolve(target, mutation_rate=0.1, generations=10)
        
        assert result1["final_fitness"] == pytest.approx(result2["final_fitness"], rel=1e-6)

"""
Weight evolution for TPN.

Phase-encoded weights can evolve to match target patterns using
a simple (1+1) evolution strategy.
"""

import math
import random
from typing import List, Dict
from .phase import PhaseWeight


class EvolvablePhaseNetwork:
    """
    A network of phase-encoded weights that can evolve.
    
    Uses a simple (1+1) evolution strategy:
    1. Create mutated offspring
    2. Compute offspring fitness
    3. Select best (parent or offspring)
    4. Repeat
    """
    
    def __init__(self, num_weights: int = 100):
        """
        Initialize evolvable phase network.
        
        Args:
            num_weights: Number of phase-encoded weights.
        """
        self.weights = [PhaseWeight(random.uniform(0, 2 * math.pi)) for _ in range(num_weights)]
        self.generation = 0
        self.best_fitness_history = []
    
    def compute_fitness(self, target: List[float]) -> float:
        """
        Compute how close the network output is to the target.
        
        Uses mean squared error converted to fitness:
        fitness = 1 / (1 + MSE)
        
        Args:
            target: Target values.
        
        Returns:
            Fitness value in [0, 1].
        """
        current = [w.value for w in self.weights[:len(target)]]
        mse = sum((c - t) ** 2 for c, t in zip(current, target)) / len(target)
        return 1.0 / (1.0 + mse)
    
    def evolve(self, target: List[float], mutation_rate: float = 0.1,
               generations: int = 50) -> Dict:
        """
        Evolve the network to match the target.
        
        Args:
            target: Target values.
            mutation_rate: Standard deviation of Gaussian mutation.
            generations: Number of generations to evolve.
        
        Returns:
            Dictionary with evolution results.
        """
        best_fitness = self.compute_fitness(target)
        self.best_fitness_history.append(best_fitness)
        
        for gen in range(generations):
            # Create mutated offspring
            offspring = [w.mutate(mutation_rate) for w in self.weights]
            
            # Compute offspring fitness
            current = [w.value for w in offspring[:len(target)]]
            mse = sum((c - t) ** 2 for c, t in zip(current, target)) / len(target)
            offspring_fitness = 1.0 / (1.0 + mse)
            
            # Select best
            if offspring_fitness > best_fitness:
                self.weights = offspring
                best_fitness = offspring_fitness
            
            self.best_fitness_history.append(best_fitness)
            self.generation += 1
        
        return {
            "final_fitness": best_fitness,
            "generations": generations,
            "fitness_history": self.best_fitness_history,
        }

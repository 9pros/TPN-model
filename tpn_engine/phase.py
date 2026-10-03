"""
Phase-encoded weights for TPN.

Weights are encoded as phase angles (radians) instead of traditional floats.
This enables:
- Continuous weight representation (not just ternary)
- Smooth fitness gradients for optimization
- Evolution through phase mutation
"""

import math
import random
from typing import List


class PhaseWeight:
    """
    A weight encoded as a phase angle.
    
    The weight value is computed as cos(phase), and the fitness is
    computed as cos²(phase - π/4).
    
    Key properties:
    - Phase 0: weight = 1.0, fitness = 0.5
    - Phase π/4: weight = 0.707, fitness = 1.0 (maximum fitness)
    - Phase π/2: weight = 0.0, fitness = 0.5
    - Phase π: weight = -1.0, fitness = 0.5
    - Phase 3π/4: weight = -0.707, fitness = 1.0 (maximum fitness)
    """
    
    def __init__(self, phase: float):
        """
        Initialize a phase weight.
        
        Args:
            phase: Phase angle in radians. Will be normalized to [0, 2π].
        """
        self.phase = phase % (2 * math.pi)
    
    @property
    def value(self) -> float:
        """Get the weight value as cos(phase)."""
        return math.cos(self.phase)
    
    @property
    def fitness(self) -> float:
        """Get the fitness as cos²(phase - π/4)."""
        return math.cos(self.phase - math.pi / 4.0) ** 2
    
    def mutate(self, mutation_rate: float = 0.1) -> 'PhaseWeight':
        """
        Create a mutated copy of this weight.
        
        Args:
            mutation_rate: Standard deviation of the Gaussian mutation.
        
        Returns:
            A new PhaseWeight with a mutated phase.
        """
        new_phase = self.phase + random.gauss(0, mutation_rate)
        return PhaseWeight(new_phase)
    
    def __repr__(self) -> str:
        return f"PhaseWeight(phase={self.phase:.4f}, value={self.value:.4f}, fitness={self.fitness:.4f})"
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, PhaseWeight):
            return False
        return self.phase == other.phase


class PhaseEncoding:
    """
    Utilities for encoding and decoding vectors/matrices as phase weights.
    """
    
    @staticmethod
    def encode_vector(values: List[float]) -> List[PhaseWeight]:
        """
        Encode a vector of floats as phase weights.
        
        The encoding maps each value to a phase angle such that
        cos(phase) = value. This means:
        - value 1.0 → phase 0
        - value 0.0 → phase π/2
        - value -1.0 → phase π
        
        Args:
            values: List of float values in [-1, 1].
        
        Returns:
            List of PhaseWeight objects.
        """
        encoded = []
        for v in values:
            # Clamp to [-1, 1] to avoid domain errors
            v = max(-1.0, min(1.0, v))
            phase = math.acos(v)
            encoded.append(PhaseWeight(phase))
        return encoded
    
    @staticmethod
    def decode_vector(weights: List[PhaseWeight]) -> List[float]:
        """
        Decode phase weights back to float values.
        
        Args:
            weights: List of PhaseWeight objects.
        
        Returns:
            List of float values.
        """
        return [w.value for w in weights]
    
    @staticmethod
    def encode_matrix(matrix: List[List[float]]) -> List[List[PhaseWeight]]:
        """
        Encode a matrix of floats as phase weights.
        
        Args:
            matrix: 2D list of float values.
        
        Returns:
            2D list of PhaseWeight objects.
        """
        return [PhaseEncoding.encode_vector(row) for row in matrix]
    
    @staticmethod
    def decode_matrix(weights: List[List[PhaseWeight]]) -> List[List[float]]:
        """
        Decode phase weights back to a matrix of floats.
        
        Args:
            weights: 2D list of PhaseWeight objects.
        
        Returns:
            2D list of float values.
        """
        return [PhaseEncoding.decode_vector(row) for row in weights]

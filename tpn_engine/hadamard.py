"""
Hadamard transform for TPN.

Uses normalized Sylvester-Walsh Hadamard transform for efficient
weight transformation. This is the same transform used by Bonsai.
"""

import math
from typing import List


class HadamardTransform:
    """
    Normalized Sylvester-Walsh Hadamard transform.
    
    The transform is its own inverse (up to normalization), making it
    efficient for forward and inverse transformations.
    """
    
    def __init__(self, block_size: int = 1024):
        """
        Initialize Hadamard transform.
        
        Args:
            block_size: Size of the transform (must be power of 2).
        """
        self.block_size = block_size
        self._matrix = None
    
    def _generate_matrix(self) -> List[List[float]]:
        """Generate Sylvester-Walsh Hadamard matrix."""
        if self._matrix is not None:
            return self._matrix
        
        n = self.block_size
        H = [[1.0]]
        size = 1
        while size < n:
            new_size = size * 2
            new_H = [[0.0] * new_size for _ in range(new_size)]
            for i in range(size):
                for j in range(size):
                    new_H[i][j] = H[i][j]
                    new_H[i][j + size] = H[i][j]
                    new_H[i + size][j] = H[i][j]
                    new_H[i + size][j + size] = -H[i][j]
            H = new_H
            size = new_size
        
        self._matrix = H
        return H
    
    def transform(self, vector: List[float]) -> List[float]:
        """
        Apply Hadamard transform to a vector.
        
        Args:
            vector: Input vector (will be padded or truncated to block_size).
        
        Returns:
            Transformed vector of length block_size.
        """
        n = len(vector)
        if n < self.block_size:
            vector = vector + [0.0] * (self.block_size - n)
        elif n > self.block_size:
            vector = vector[:self.block_size]
        
        H = self._generate_matrix()
        result = [0.0] * self.block_size
        for i in range(self.block_size):
            for j in range(self.block_size):
                result[i] += H[i][j] * vector[j]
        
        # Normalize
        norm = math.sqrt(self.block_size)
        return [r / norm for r in result]
    
    def inverse_transform(self, vector: List[float]) -> List[float]:
        """
        Apply inverse Hadamard transform.
        
        For Hadamard transform, the inverse is the same as the forward
        transform (up to normalization).
        
        Args:
            vector: Input vector.
        
        Returns:
            Inverse transformed vector.
        """
        return self.transform(vector)

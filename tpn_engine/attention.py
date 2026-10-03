"""
TPN Attention mechanism.

Uses fitness-based softmax instead of standard exp-based softmax.
The fitness function cos²(θ - π/4) provides smooth, bounded weights.
"""

import math
from typing import List


def tpn_softmax(scores: List[List[float]]) -> List[List[float]]:
    """
    Compute softmax using trigonometric fitness function.
    
    Instead of exp(x_i) / sum(exp(x_j)), uses:
    cos²(θ_i - π/4) / sum(cos²(θ_j - π/4))
    
    where θ_i = π/2 * (x_i - min) / (max - min)
    
    Args:
        scores: 2D list of attention scores.
    
    Returns:
        2D list of attention weights (probabilities).
    """
    weights = [[0.0] * len(scores[0]) for _ in range(len(scores))]
    
    for i in range(len(scores)):
        min_score = min(scores[i])
        max_score = max(scores[i])
        
        fitness_values = []
        for j in range(len(scores[i])):
            if max_score > min_score:
                theta = math.pi / 2.0 * (scores[i][j] - min_score) / (max_score - min_score)
            else:
                theta = math.pi / 4.0
            fitness = math.cos(theta - math.pi / 4.0) ** 2
            fitness_values.append(fitness)
        
        total = sum(fitness_values)
        for j in range(len(scores[i])):
            weights[i][j] = fitness_values[j] / total
    
    return weights


def standard_softmax(scores: List[List[float]]) -> List[List[float]]:
    """
    Compute standard softmax using exp function.
    
    Args:
        scores: 2D list of attention scores.
    
    Returns:
        2D list of attention weights (probabilities).
    """
    weights = [[0.0] * len(scores[0]) for _ in range(len(scores))]
    
    for i in range(len(scores)):
        max_score = max(scores[i])
        exp_scores = [math.exp(s - max_score) for s in scores[i]]
        total = sum(exp_scores)
        for j in range(len(scores[i])):
            weights[i][j] = exp_scores[j] / total
    
    return weights


class TPNAttention:
    """
    TPN Attention mechanism.
    
    Uses fitness-based softmax for attention weights.
    """
    
    @staticmethod
    def compute_scores(q: List[List[float]], k: List[List[float]]) -> List[List[float]]:
        """
        Compute attention scores: Q * K^T / sqrt(d)
        
        Args:
            q: Query matrix (n x d)
            k: Key matrix (m x d)
        
        Returns:
            Score matrix (n x m)
        """
        d = len(q[0])
        sqrt_d = math.sqrt(d)
        n = len(q)
        m = len(k)
        
        scores = [[0.0] * m for _ in range(n)]
        for i in range(n):
            for j in range(m):
                s = sum(q[i][d_idx] * k[j][d_idx] for d_idx in range(d))
                scores[i][j] = s / sqrt_d
        
        return scores
    
    @staticmethod
    def compute_output(weights: List[List[float]], v: List[List[float]]) -> List[List[float]]:
        """
        Compute output: attention_weights * V
        
        Args:
            weights: Attention weights (n x m)
            v: Value matrix (m x d)
        
        Returns:
            Output matrix (n x d)
        """
        n = len(weights)
        d = len(v[0])
        output = [[0.0] * d for _ in range(n)]
        
        for i in range(n):
            for j in range(d):
                output[i][j] = sum(weights[i][k] * v[k][j] for k in range(len(weights[i])))
        
        return output
    
    @staticmethod
    def forward(q: List[List[float]], k: List[List[float]], v: List[List[float]]) -> List[List[float]]:
        """
        Full TPN attention forward pass.
        
        Args:
            q: Query matrix (n x d)
            k: Key matrix (m x d)
            v: Value matrix (m x d)
        
        Returns:
            Output matrix (n x d)
        """
        scores = TPNAttention.compute_scores(q, k)
        weights = tpn_softmax(scores)
        return TPNAttention.compute_output(weights, v)
    
    @staticmethod
    def forward_standard(q: List[List[float]], k: List[List[float]], v: List[List[float]]) -> List[List[float]]:
        """
        Standard attention forward pass (for comparison).
        
        Args:
            q: Query matrix (n x d)
            k: Key matrix (m x d)
            v: Value matrix (m x d)
        
        Returns:
            Output matrix (n x d)
        """
        scores = TPNAttention.compute_scores(q, k)
        weights = standard_softmax(scores)
        return TPNAttention.compute_output(weights, v)

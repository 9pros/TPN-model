"""
Full TPN Inference Engine.

A complete inference engine that uses phase-based computation for
neural network operations. This is the core of the TPN system.
"""

import math
import random
from typing import List, Dict, Optional
from .phase import PhaseWeight, PhaseEncoding
from .attention import TPNAttention, tpn_softmax
from .hadamard import HadamardTransform
from .evolution import EvolvablePhaseNetwork


class TPNConfig:
    """Configuration for TPN model."""
    
    def __init__(self, hidden_size: int = 512, num_layers: int = 4,
                 num_heads: int = 4, head_dim: int = 128,
                 intermediate_size: int = 2048, max_position_embeddings: int = 2048,
                 vocab_size: int = 32000):
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.vocab_size = vocab_size


class TPNModel:
    """
    TPN Inference Model.
    
    A neural network that uses phase-based computation for all operations:
    - Phase-encoded weights
    - Fitness-based attention
    - Hadamard transform for efficient computation
    - Evolution for weight optimization
    """
    
    def __init__(self, config: TPNConfig):
        self.config = config
        self.hadamard = HadamardTransform(block_size=1024)
        self.layers = self._init_layers()
    
    def _init_layers(self) -> List[Dict]:
        """Initialize model layers with random phase-encoded weights."""
        layers = []
        for i in range(self.config.num_layers):
            layer = {
                "index": i,
                "q_proj": self._random_phase_matrix(self.config.hidden_size, self.config.hidden_size),
                "k_proj": self._random_phase_matrix(self.config.hidden_size, self.config.hidden_size),
                "v_proj": self._random_phase_matrix(self.config.hidden_size, self.config.hidden_size),
                "o_proj": self._random_phase_matrix(self.config.hidden_size, self.config.hidden_size),
                "gate_proj": self._random_phase_matrix(self.config.intermediate_size, self.config.hidden_size),
                "up_proj": self._random_phase_matrix(self.config.intermediate_size, self.config.hidden_size),
                "down_proj": self._random_phase_matrix(self.config.hidden_size, self.config.intermediate_size),
            }
            layers.append(layer)
        return layers
    
    def _random_phase_matrix(self, rows: int, cols: int) -> List[List[PhaseWeight]]:
        """Create a matrix of random phase-encoded weights."""
        return [[PhaseWeight(random.uniform(0, 2 * math.pi)) for _ in range(cols)] for _ in range(rows)]
    
    def _matvec(self, matrix: List[List[PhaseWeight]], vector: List[float]) -> List[float]:
        """Matrix-vector multiplication with phase-encoded weights."""
        result = [0.0] * len(matrix)
        for i in range(len(matrix)):
            s = 0.0
            for j in range(len(vector)):
                s += matrix[i][j].value * vector[j]
            result[i] = s
        return result
    
    def _attention(self, q: List[float], k: List[float], v: List[float]) -> List[float]:
        """Single-head attention using TPN fitness-based softmax."""
        # Compute score
        d = len(q)
        sqrt_d = math.sqrt(d)
        score = sum(q[i] * k[i] for i in range(d)) / sqrt_d
        
        # TPN fitness
        fitness = math.cos(score - math.pi / 4.0) ** 2
        
        # Weighted sum
        return [fitness * v[i] for i in range(len(v))]
    
    def _ffn(self, x: List[float], layer: Dict) -> List[float]:
        """Feed-forward network with phase-encoded weights."""
        # Gate and up projections
        gate = self._matvec(layer["gate_proj"], x)
        up = self._matvec(layer["up_proj"], x)
        
        # SiLU activation: x * sigmoid(x) (numerically stable)
        def silu(x: float) -> float:
            if x >= 0:
                return x / (1.0 + math.exp(-x))
            else:
                ex = math.exp(x)
                return x * ex / (1.0 + ex)
        
        ffn = [silu(g) * u for g, u in zip(gate, up)]
        
        # Down projection
        return self._matvec(layer["down_proj"], ffn)
    
    def forward(self, input_data: List[float]) -> List[float]:
        """
        Forward pass through the TPN model.
        
        Args:
            input_data: Input vector of size hidden_size.
        
        Returns:
            Output vector of size hidden_size.
        """
        hidden = input_data[:self.config.hidden_size]
        
        for layer in self.layers:
            # Self-attention
            q = self._matvec(layer["q_proj"], hidden)
            k = self._matvec(layer["k_proj"], hidden)
            v = self._matvec(layer["v_proj"], hidden)
            
            # Attention output
            attn_out = self._attention(q, k, v)
            
            # Output projection
            hidden = self._matvec(layer["o_proj"], attn_out)
            
            # FFN
            hidden = self._ffn(hidden, layer)
        
        return hidden
    
    def evolve(self, target: List[float], generations: int = 50,
               mutation_rate: float = 0.1) -> Dict:
        """
        Evolve the model to match target output.
        
        Args:
            target: Target output vector.
            generations: Number of generations to evolve.
            mutation_rate: Standard deviation of Gaussian mutation.
        
        Returns:
            Dictionary with evolution results.
        """
        # Create evolvable network for each layer
        networks = []
        for layer in self.layers:
            network = EvolvablePhaseNetwork(num_weights=self.config.hidden_size)
            networks.append(network)
        
        # Evolve each layer
        results = []
        for i, network in enumerate(networks):
            result = network.evolve(target, mutation_rate=mutation_rate, generations=generations)
            results.append(result)
        
        return {
            "generations": generations,
            "final_fitness": results[-1]["final_fitness"],
            "layer_results": results,
        }
    
    def get_config(self) -> Dict:
        """Get model configuration."""
        return {
            "hidden_size": self.config.hidden_size,
            "num_layers": self.config.num_layers,
            "num_heads": self.config.num_heads,
            "head_dim": self.config.head_dim,
            "intermediate_size": self.config.intermediate_size,
            "max_position_embeddings": self.config.max_position_embeddings,
            "vocab_size": self.config.vocab_size,
        }

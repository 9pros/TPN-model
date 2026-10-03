"""
Trajectory prediction for TPN.

Simulates multiple possible futures (trajectories) and picks the best one
based on fitness. This is the key advantage of TPN: we can explore many
paths in parallel and select the optimal one.
"""

import math
import random
from typing import List, Dict, Optional, Callable
from dataclasses import dataclass


@dataclass
class Trajectory:
    """A predicted trajectory with its fitness score."""
    path: List[float]
    fitness: float
    metadata: Optional[Dict] = None


class TrajectoryPredictor:
    """
    Predicts multiple trajectories and selects the best one.
    
    Uses phase-based mutation to explore different paths through
    the model's latent space.
    """
    
    def __init__(self, num_trajectories: int = 10, hidden_size: int = 128,
                 mutation_rate: float = 0.1, steps: int = 5):
        """
        Initialize trajectory predictor.
        
        Args:
            num_trajectories: Number of trajectories to simulate.
            hidden_size: Size of the hidden state.
            mutation_rate: Rate of phase mutation for exploration.
            steps: Number of steps to predict ahead.
        """
        self.num_trajectories = num_trajectories
        self.hidden_size = hidden_size
        self.mutation_rate = mutation_rate
        self.steps = steps
    
    def _mutate_input(self, input_data: List[float]) -> List[float]:
        """Create a mutated copy of the input."""
        mutated = []
        for val in input_data:
            # Add small random perturbation
            noise = random.gauss(0, self.mutation_rate)
            mutated.append(val + noise)
        return mutated
    
    def _compute_trajectory_fitness(self, path: List[float], target: Optional[List[float]] = None) -> float:
        """
        Compute fitness of a trajectory.
        
        If target is provided, fitness is based on distance to target.
        Otherwise, fitness is based on the "smoothness" of the trajectory.
        """
        if target is not None:
            # Distance to target (closer is better)
            if len(path) != len(target):
                return 0.0
            mse = sum((p - t) ** 2 for p, t in zip(path, target)) / len(path)
            return 1.0 / (1.0 + mse)
        else:
            # Smoothness (smaller changes are better)
            if len(path) < 2:
                return 1.0
            total_variation = sum(abs(path[i] - path[i-1]) for i in range(1, len(path)))
            avg_variation = total_variation / (len(path) - 1)
            return 1.0 / (1.0 + avg_variation)
    
    def predict(self, model, input_data: List[float], target: Optional[List[float]] = None) -> List[Trajectory]:
        """
        Predict multiple trajectories and return them sorted by fitness.
        
        Args:
            model: TPN model to use for prediction.
            input_data: Current state.
            target: Optional target state to aim for.
        
        Returns:
            List of trajectories sorted by fitness (best first).
        """
        trajectories = []
        
        for i in range(self.num_trajectories):
            # Create mutated input for this trajectory
            mutated_input = self._mutate_input(input_data)
            
            # Run model forward
            output = model.forward(mutated_input)
            
            # Compute fitness
            fitness = self._compute_trajectory_fitness(output, target)
            
            # Create trajectory
            traj = Trajectory(
                path=output,
                fitness=fitness,
                metadata={"trajectory_id": i, "mutation_rate": self.mutation_rate}
            )
            trajectories.append(traj)
        
        # Sort by fitness (best first)
        trajectories.sort(key=lambda t: t.fitness, reverse=True)
        
        return trajectories
    
    def predict_best(self, model, input_data: List[float], target: Optional[List[float]] = None) -> Trajectory:
        """
        Predict and return only the best trajectory.
        
        Args:
            model: TPN model to use for prediction.
            input_data: Current state.
            target: Optional target state to aim for.
        
        Returns:
            The best trajectory.
        """
        trajectories = self.predict(model, input_data, target)
        return trajectories[0] if trajectories else None
    
    def predict_with_evolution(self, model, input_data: List[float], target: List[float],
                               generations: int = 10) -> List[Trajectory]:
        """
        Predict trajectories and evolve them toward the target.
        
        This is the key TPN advantage: we can evolve trajectories
        to find better solutions.
        
        Args:
            model: TPN model to use.
            input_data: Current state.
            target: Target state.
            generations: Number of evolution generations.
        
        Returns:
            List of evolved trajectories sorted by fitness.
        """
        # Initial prediction
        trajectories = self.predict(model, input_data, target)
        
        # Evolve each trajectory
        for gen in range(generations):
            new_trajectories = []
            
            for traj in trajectories:
                # Mutate the path
                mutated_path = []
                for val in traj.path:
                    noise = random.gauss(0, self.mutation_rate * 0.5)
                    mutated_path.append(val + noise)
                
                # Compute new fitness
                new_fitness = self._compute_trajectory_fitness(mutated_path, target)
                
                # Keep if better
                if new_fitness > traj.fitness:
                    new_trajectories.append(Trajectory(
                        path=mutated_path,
                        fitness=new_fitness,
                        metadata={**traj.metadata, "generation": gen}
                    ))
                else:
                    new_trajectories.append(traj)
            
            trajectories = new_trajectories
        
        # Sort by fitness
        trajectories.sort(key=lambda t: t.fitness, reverse=True)
        
        return trajectories

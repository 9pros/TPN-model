"""
Tests for trajectory prediction.
"""

import math
import pytest
from tpn_engine.trajectory import TrajectoryPredictor, Trajectory


class TestTrajectory:
    """Test trajectory prediction."""
    
    def test_trajectory_creation(self):
        """Test creating a trajectory."""
        path = [0.0, 0.1, 0.2, 0.3]
        fitness = 0.8
        traj = Trajectory(path=path, fitness=fitness)
        assert traj.path == path
        assert traj.fitness == fitness
    
    def test_trajectory_predictor_creation(self):
        """Test creating a trajectory predictor."""
        predictor = TrajectoryPredictor(num_trajectories=5, hidden_size=32)
        assert predictor.num_trajectories == 5
        assert predictor.hidden_size == 32
    
    def test_predict_returns_multiple_trajectories(self):
        """Test that predict returns multiple trajectories."""
        from tpn_engine import TPNModel, TPNConfig
        config = TPNConfig(hidden_size=32, num_layers=1, num_heads=2, head_dim=16)
        model = TPNModel(config)
        predictor = TrajectoryPredictor(num_trajectories=5, hidden_size=32)
        
        input_data = [0.1] * 32
        trajectories = predictor.predict(model, input_data)
        
        assert len(trajectories) == 5
    
    def test_predict_returns_sorted_trajectories(self):
        """Test that trajectories are sorted by fitness (best first)."""
        from tpn_engine import TPNModel, TPNConfig
        config = TPNConfig(hidden_size=32, num_layers=1, num_heads=2, head_dim=16)
        model = TPNModel(config)
        predictor = TrajectoryPredictor(num_trajectories=5, hidden_size=32)
        
        input_data = [0.1] * 32
        trajectories = predictor.predict(model, input_data)
        
        # Check sorted by fitness (descending)
        for i in range(len(trajectories) - 1):
            assert trajectories[i].fitness >= trajectories[i + 1].fitness
    
    def test_predict_best_trajectory(self):
        """Test getting the best trajectory."""
        from tpn_engine import TPNModel, TPNConfig
        config = TPNConfig(hidden_size=32, num_layers=1, num_heads=2, head_dim=16)
        model = TPNModel(config)
        predictor = TrajectoryPredictor(num_trajectories=5, hidden_size=32)
        
        input_data = [0.1] * 32
        best = predictor.predict_best(model, input_data)
        
        assert best is not None
        assert hasattr(best, 'path')
        assert hasattr(best, 'fitness')
    
    def test_trajectory_fitness_in_range(self):
        """Test that trajectory fitness is in [0, 1]."""
        from tpn_engine import TPNModel, TPNConfig
        config = TPNConfig(hidden_size=32, num_layers=1, num_heads=2, head_dim=16)
        model = TPNModel(config)
        predictor = TrajectoryPredictor(num_trajectories=5, hidden_size=32)
        
        input_data = [0.1] * 32
        trajectories = predictor.predict(model, input_data)
        
        for traj in trajectories:
            assert 0 <= traj.fitness <= 1

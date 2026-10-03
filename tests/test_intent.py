"""
Tests for intent prediction.
"""

import math
import pytest
from tpn_engine.intent import IntentPredictor, IntentState


class TestIntentState:
    """Test intent state."""
    
    def test_intent_state_creation(self):
        """Test creating an intent state."""
        state = IntentState(
            current_intent=0.5,
            confidence=0.8,
            history=[0.3, 0.4, 0.5]
        )
        assert state.current_intent == 0.5
        assert state.confidence == 0.8
        assert state.history == [0.3, 0.4, 0.5]
    
    def test_intent_state_update(self):
        """Test updating intent state."""
        state = IntentState(current_intent=0.5, confidence=0.8)
        new_state = state.update(0.6, 0.9)
        assert new_state.current_intent == 0.6
        assert new_state.confidence == 0.9
        assert len(new_state.history) == 1


class TestIntentPredictor:
    """Test intent prediction."""
    
    def test_predictor_creation(self):
        """Test creating an intent predictor."""
        predictor = IntentPredictor(hidden_size=32, num_intents=5)
        assert predictor.hidden_size == 32
        assert predictor.num_intents == 5
    
    def test_predict_returns_intent_state(self):
        """Test that predict returns an intent state."""
        from tpn_engine import TPNModel, TPNConfig
        config = TPNConfig(hidden_size=32, num_layers=1, num_heads=2, head_dim=16)
        model = TPNModel(config)
        predictor = IntentPredictor(hidden_size=32, num_intents=5)
        
        input_data = [0.1] * 32
        state = predictor.predict(model, input_data)
        
        assert isinstance(state, IntentState)
        assert 0 <= state.current_intent <= 1
        assert 0 <= state.confidence <= 1
    
    def test_predict_with_history(self):
        """Test prediction with history."""
        from tpn_engine import TPNModel, TPNConfig
        config = TPNConfig(hidden_size=32, num_layers=1, num_heads=2, head_dim=16)
        model = TPNModel(config)
        predictor = IntentPredictor(hidden_size=32, num_intents=5)
        
        input_data = [0.1] * 32
        state1 = predictor.predict(model, input_data)
        state2 = predictor.predict(model, input_data, previous_state=state1)
        
        assert len(state2.history) > len(state1.history)
    
    def test_predict_continuous(self):
        """Test continuous intent prediction."""
        from tpn_engine import TPNModel, TPNConfig
        config = TPNConfig(hidden_size=32, num_layers=1, num_heads=2, head_dim=16)
        model = TPNModel(config)
        predictor = IntentPredictor(hidden_size=32, num_intents=5)
        
        # Simulate continuous prediction
        states = []
        for i in range(10):
            input_data = [0.1 * i] * 32
            if states:
                state = predictor.predict(model, input_data, previous_state=states[-1])
            else:
                state = predictor.predict(model, input_data)
            states.append(state)
        
        # Check that intent evolves
        intents = [s.current_intent for s in states]
        # Intent should change over time
        assert len(set(intents)) > 1
    
    def test_evolve_intent(self):
        """Test evolving intent toward target."""
        from tpn_engine import TPNModel, TPNConfig
        config = TPNConfig(hidden_size=32, num_layers=1, num_heads=2, head_dim=16)
        model = TPNModel(config)
        predictor = IntentPredictor(hidden_size=32, num_intents=5)
        
        input_data = [0.1] * 32
        target_intent = 0.8
        
        result = predictor.evolve_intent(model, input_data, target_intent, generations=20)
        
        assert "final_intent" in result
        assert "final_confidence" in result
        assert "generations" in result
        assert result["generations"] == 20
    
    def test_intent_confidence_increases_with_evolution(self):
        """Test that confidence increases with evolution."""
        from tpn_engine import TPNModel, TPNConfig
        config = TPNConfig(hidden_size=32, num_layers=1, num_heads=2, head_dim=16)
        model = TPNModel(config)
        predictor = IntentPredictor(hidden_size=32, num_intents=5)
        
        input_data = [0.1] * 32
        target_intent = 0.8
        
        result = predictor.evolve_intent(model, input_data, target_intent, generations=50)
        
        # Confidence should increase
        assert result["final_confidence"] > result["initial_confidence"]

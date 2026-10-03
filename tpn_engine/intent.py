"""
Intent prediction for TPN.

Continuously predicts user intent and evolves toward target intent.
This enables the model to "understand" what the user wants and
adapt its behavior accordingly.
"""

import math
import random
from typing import List, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class IntentState:
    """Current intent state with confidence and history."""
    current_intent: float
    confidence: float
    history: List[float] = field(default_factory=list)
    
    def update(self, new_intent: float, new_confidence: float) -> 'IntentState':
        """Create updated intent state."""
        new_history = self.history + [self.current_intent]
        # Keep only last 10 intents
        new_history = new_history[-10:]
        return IntentState(
            current_intent=new_intent,
            confidence=new_confidence,
            history=new_history
        )


class IntentPredictor:
    """
    Predicts and evolves user intent.
    
    Uses phase-based computation to continuously track intent
    and evolve toward target intent.
    """
    
    def __init__(self, hidden_size: int = 128, num_intents: int = 10,
                 evolution_rate: float = 0.1):
        """
        Initialize intent predictor.
        
        Args:
            hidden_size: Size of hidden state.
            num_intents: Number of possible intents.
            evolution_rate: Rate of intent evolution.
        """
        self.hidden_size = hidden_size
        self.num_intents = num_intents
        self.evolution_rate = evolution_rate
    
    def _compute_intent(self, output: List[float]) -> float:
        """
        Compute intent from model output.
        
        Uses the first element of output as intent signal.
        """
        if not output:
            return 0.0
        # Use first element, normalized to [0, 1]
        raw_intent = output[0]
        # Numerically stable sigmoid
        if raw_intent >= 0:
            return 1.0 / (1.0 + math.exp(-raw_intent))
        else:
            ex = math.exp(raw_intent)
            return ex / (1.0 + ex)
    
    def _compute_confidence(self, output: List[float]) -> float:
        """
        Compute confidence from model output.
        
        Uses the entropy of the output distribution as confidence signal.
        Higher entropy = lower confidence (more uncertain).
        Lower entropy = higher confidence (more certain).
        """
        if len(output) < 2:
            return 0.5
        
        # Normalize to probabilities
        total = sum(abs(x) for x in output)
        if total < 1e-10:
            return 0.5
        
        probs = [abs(x) / total for x in output]
        
        # Compute entropy
        entropy = 0.0
        for p in probs:
            if p > 1e-10:
                entropy -= p * math.log(p)
        
        # Normalize entropy to [0, 1] (max entropy = log(n))
        max_entropy = math.log(len(output))
        if max_entropy < 1e-10:
            return 0.5
        
        normalized_entropy = entropy / max_entropy
        
        # Confidence = 1 - entropy (lower entropy = higher confidence)
        return 1.0 - normalized_entropy
    
    def predict(self, model, input_data: List[float],
                previous_state: Optional[IntentState] = None) -> IntentState:
        """
        Predict current intent.
        
        Args:
            model: TPN model.
            input_data: Current input.
            previous_state: Previous intent state (for continuity).
        
        Returns:
            Current intent state.
        """
        # Run model
        output = model.forward(input_data)
        
        # Compute intent and confidence
        intent = self._compute_intent(output)
        confidence = self._compute_confidence(output)
        
        # If we have previous state, blend with it
        if previous_state is not None:
            # Exponential moving average
            alpha = 0.7  # Weight for new intent
            blended_intent = alpha * intent + (1 - alpha) * previous_state.current_intent
            blended_confidence = alpha * confidence + (1 - alpha) * previous_state.confidence
            return previous_state.update(blended_intent, blended_confidence)
        else:
            return IntentState(current_intent=intent, confidence=confidence)
    
    def evolve_intent(self, model, input_data: List[float], target_intent: float,
                      generations: int = 50) -> Dict:
        """
        Evolve intent toward target.
        
        This is the key TPN advantage: we can evolve the model's
        intent to match what the user wants.
        
        Args:
            model: TPN model.
            input_data: Initial input.
            target_intent: Target intent value.
            generations: Number of evolution generations.
        
        Returns:
            Dictionary with evolution results.
        """
        # Initial prediction
        state = self.predict(model, input_data)
        initial_intent = state.current_intent
        initial_confidence = state.confidence
        
        best_intent = initial_intent
        best_confidence = initial_confidence
        best_distance = abs(initial_intent - target_intent)
        
        for gen in range(generations):
            # Mutate input
            mutated_input = []
            for val in input_data:
                noise = random.gauss(0, self.evolution_rate)
                mutated_input.append(val + noise)
            
            # Predict with mutated input
            new_state = self.predict(model, mutated_input, state)
            
            # Check if closer to target
            new_distance = abs(new_state.current_intent - target_intent)
            if new_distance < best_distance:
                best_intent = new_state.current_intent
                best_confidence = new_state.confidence
                best_distance = new_distance
                state = new_state
        
        return {
            "initial_intent": initial_intent,
            "final_intent": best_intent,
            "initial_confidence": initial_confidence,
            "final_confidence": best_confidence,
            "target_intent": target_intent,
            "generations": generations,
            "final_distance": best_distance,
        }

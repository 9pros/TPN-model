"""
Tests for echo signal and echo-driven evolution.

An "echo" is the model's own output fed back as input after a delay,
creating a resonance loop. Evolution selects phase configurations that
maximize echo coherence (constructive interference over time).
"""

import math
import pytest
from tpn_engine.echo import EchoSignal, EchoEvolution


class TestEchoSignal:
    def test_creation(self):
        echo = EchoSignal(delay=1.0, decay=0.5)
        assert echo.delay == 1.0
        assert echo.decay == 0.5

    def test_single_echo(self):
        echo = EchoSignal(delay=1.0, decay=0.5)
        echo.inject([1.0, 2.0, 3.0], at_time=0.0)
        out = echo.sample(at_time=1.0)
        # at exactly one delay, should be decay * original
        assert abs(out[0] - 0.5) < 1e-6
        assert abs(out[1] - 1.0) < 1e-6
        assert abs(out[2] - 1.5) < 1e-6

    def test_echo_decays(self):
        echo = EchoSignal(delay=1.0, decay=0.5)
        echo.inject([1.0], at_time=0.0)
        o1 = echo.sample(at_time=1.0)[0]
        o2 = echo.sample(at_time=2.0)[0]
        assert o2 < o1, "echo should decay over time"

    def test_multiple_injections_superpose(self):
        echo = EchoSignal(delay=1.0, decay=0.5)
        echo.inject([1.0], at_time=0.0)
        echo.inject([1.0], at_time=0.0)
        out = echo.sample(at_time=1.0)[0]
        # two identical injections superpose
        assert abs(out - 1.0) < 1e-6

    def test_coherence_metric(self):
        echo = EchoSignal(delay=1.0, decay=0.9)
        # Constructive: inject same phase repeatedly
        for i in range(10):
            echo.inject([1.0, 1.0], at_time=float(i))
        c = echo.coherence()
        assert 0.0 <= c <= 1.0

    def test_coherence_constructive_vs_destructive(self):
        """Constructive (aligned) echoes must out-cohere destructive (alternating)."""
        good = EchoSignal(delay=1.0, decay=0.9)
        bad = EchoSignal(delay=1.0, decay=0.9)
        for i in range(10):
            good.inject([1.0, 1.0], at_time=float(i))
            # alternate sign every round -> 180 degree phase flips -> cancellation
            sign = 1.0 if i % 2 == 0 else -1.0
            bad.inject([sign, sign], at_time=float(i))
        assert good.coherence() > bad.coherence()


class TestEchoEvolution:
    def test_creation(self):
        evo = EchoEvolution(num_phases=16, generations=5)
        assert evo.num_phases == 16
        assert evo.generations == 5

    def test_evolve_improves_coherence(self):
        evo = EchoEvolution(num_phases=16, generations=20)
        result = evo.evolve()
        assert result["final_coherence"] >= result["initial_coherence"]

    def test_evolve_returns_history(self):
        evo = EchoEvolution(num_phases=16, generations=10)
        result = evo.evolve()
        assert "history" in result
        assert len(result["history"]) == 10

    def test_best_phases_are_bounded(self):
        evo = EchoEvolution(num_phases=16, generations=10)
        result = evo.evolve()
        for p in result["best_phases"]:
            assert 0.0 <= p <= 2 * math.pi

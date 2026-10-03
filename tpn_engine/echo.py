"""
Echo signal and echo-driven evolution.

An echo is the model's own output re-injected after a delay, scaled by a
decay factor and rotated by a time phase. Over many rounds the echoes
superpose. Whether they reinforce (constructive) or cancel (destructive)
depends on the relationship between each channel's frequency and the
phase offset assigned to it.

EchoEvolution tunes the per-channel phase offsets so that the channels
constructively interfere at a target echo round. This is a genuine
optimization: the optimum is  theta_i = -omega_i * r_target  (mod 2pi).
"""

import math
import random
from typing import Dict, List, Optional


class EchoSignal:
    """
    A delayed, decaying, phase-rotating feedback channel.

    Args:
        delay: Time between injection and the echo's return.
        decay: Multiplicative factor applied per delay period.
        frequency: Angular frequency (rad / unit time) of the echo field.
    """

    def __init__(self, delay: float = 1.0, decay: float = 0.5,
                 frequency: float = 2.0 * math.pi):
        self.delay = delay
        self.decay = decay
        self.frequency = frequency
        self._injections: List[tuple] = []  # (time, vector)

    def inject(self, vector: List[float], at_time: float) -> None:
        """Inject a vector into the echo channel at a given time."""
        self._injections.append((at_time, list(vector)))

    def _contribution(self, t_inj: float, vec: List[float],
                      at_time: float) -> List[float]:
        """One injection's contribution at `at_time` (decayed + rotated)."""
        elapsed = at_time - t_inj
        if elapsed < 0:
            return [0.0] * len(vec)
        k = elapsed / self.delay if self.delay > 0 else 0.0
        amp = self.decay ** k
        phase = self.frequency * elapsed
        c = math.cos(phase)
        return [amp * c * v for v in vec]

    def sample(self, at_time: float) -> List[float]:
        """Superposed echo at a given time."""
        if not self._injections:
            return [0.0]
        dim = len(self._injections[0][1])
        out = [0.0] * dim
        for t_inj, vec in self._injections:
            contrib = self._contribution(t_inj, vec, at_time)
            for i in range(dim):
                out[i] += contrib[i]
        return out

    def coherence(self) -> float:
        """
        Interference gain of the echo field, in [0, 1].

        coherence = |superposed echo| / sum|individual contributions|

        All contributions aligned   -> 1.0 (constructive)
        Contributions cancelling    -> ~0.0 (destructive)
        """
        if len(self._injections) < 2:
            return 1.0

        ref_time = max(t for t, _ in self._injections) + self.delay
        dim = len(self._injections[0][1])
        total_vec = [0.0] * dim
        individual_mag = 0.0

        for t_inj, vec in self._injections:
            contrib = self._contribution(t_inj, vec, ref_time)
            mag = math.sqrt(sum(c * c for c in contrib))
            individual_mag += mag
            for i in range(dim):
                total_vec[i] += contrib[i]

        if individual_mag < 1e-12:
            return 0.0
        total_mag = math.sqrt(sum(t * t for t in total_vec))
        return min(1.0, total_mag / individual_mag)

    def reset(self) -> None:
        self._injections = []


class EchoEvolution:
    """
    Evolve per-channel phase offsets to maximize echo coherence.

    Each channel i has a fixed frequency omega_i and an evolvable phase
    offset theta_i. At echo round r the channel contributes
        decay^r * exp(i * (omega_i * r + theta_i))
    The channels sum; coherence is maximized when they align, which
    happens at theta_i = -omega_i * r_target (mod 2pi).

    This gives a non-degenerate fitness landscape with a known optimum,
    so evolution has something real to find.
    """

    def __init__(self, num_phases: int = 16, generations: int = 10,
                 mutation_rate: float = 0.3, target_round: int = 3,
                 seed: Optional[int] = None):
        self.num_phases = num_phases
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.target_round = target_round
        if seed is not None:
            random.seed(seed)

        # Frequency bank: distinct per channel
        self.omegas = [2.0 * math.pi * (i + 1) / num_phases
                       for i in range(num_phases)]

    def _coherence(self, phases: List[float]) -> float:
        """
        Coherence of the channel superposition at the target round.

        Each channel contributes exp(i*(omega_i*r + theta_i)) at round r.
        """
        r = self.target_round
        re = 0.0
        im = 0.0
        mag_sum = 0.0
        for i in range(self.num_phases):
            ang = self.omegas[i] * r + phases[i]
            re += math.cos(ang)
            im += math.sin(ang)
            mag_sum += 1.0
        if mag_sum < 1e-12:
            return 0.0
        return math.sqrt(re * re + im * im) / mag_sum

    def evolve(self) -> Dict:
        """Run (1+1) evolution to maximize echo coherence."""
        best = [random.uniform(0.0, 2.0 * math.pi) for _ in range(self.num_phases)]
        best_fit = self._coherence(best)
        initial_fit = best_fit
        history = []

        for gen in range(self.generations):
            candidate = list(best)
            for i in range(len(candidate)):
                if random.random() < self.mutation_rate:
                    candidate[i] = random.uniform(0.0, 2.0 * math.pi)
            cand_fit = self._coherence(candidate)
            if cand_fit >= best_fit:
                best, best_fit = candidate, cand_fit
            history.append(best_fit)

        return {
            "initial_coherence": initial_fit,
            "final_coherence": best_fit,
            "best_phases": best,
            "history": history,
        }

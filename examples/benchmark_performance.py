"""
Baseline performance benchmark for the TPN engine hot paths.

Measures the pure-Python implementations before optimization, so the
speedup from the NumPy rewrite can be reported honestly.

Run:  python3 examples/benchmark_performance.py
"""

import sys
import os
import time
import math
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tpn_engine.phase import PhaseWeight
from tpn_engine.hadamard import HadamardTransform
from tpn_engine.attention import TPNAttention
from tpn_engine.inference import TPNModel, TPNConfig


def hr(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def timeit(fn, repeat=3):
    """Best-of-N wall time in seconds."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def bench_matvec():
    hr("1. MATVEC (phase-encoded weight matrix @ vector)")

    random.seed(1)
    print(f"{'size':>10}{'time (ms)':>14}{'GFLOP/s':>12}")
    print("-" * 36)

    for n in [64, 128, 256, 512]:
        matrix = [[PhaseWeight(math.pi / 2 + random.gauss(0, 0.1))
                   for _ in range(n)] for _ in range(n)]
        vector = [random.uniform(-1, 1) for _ in range(n)]

        def run():
            result = [0.0] * n
            for i in range(n):
                s = 0.0
                row = matrix[i]
                for j in range(n):
                    s += row[j].value * vector[j]
                result[i] = s
            return result

        t = timeit(run)
        flops = 2 * n * n
        print(f"{n:>10}{t*1000:>14.3f}{flops/t/1e9:>12.4f}")


def bench_attention():
    hr("2. ATTENTION (Q@K^T/sqrt(d) -> fitness softmax -> @V)")

    random.seed(2)
    print(f"{'n x m x d':>16}{'time (ms)':>14}")
    print("-" * 30)

    for n, m, d in [(4, 4, 64), (8, 8, 128), (16, 16, 256), (32, 32, 256)]:
        q = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
        k = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(m)]
        v = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(m)]

        t = timeit(lambda: TPNAttention.forward(q, k, v))
        print(f"{n}x{m}x{d:<8}{t*1000:>14.3f}")


def bench_hadamard():
    hr("3. HADAMARD TRANSFORM (matrix-based, block_size)")

    print(f"{'block_size':>12}{'time (ms)':>14}")
    print("-" * 26)

    for bs in [64, 128, 256, 512]:
        ht = HadamardTransform(block_size=bs)
        vector = [random.uniform(-1, 1) for _ in range(bs)]

        t = timeit(lambda: ht.transform(vector), repeat=2)
        print(f"{bs:>12}{t*1000:>14.3f}")


def bench_forward():
    hr("4. FULL MODEL FORWARD PASS")

    print(f"{'hidden':>8}{'layers':>8}{'time (ms)':>14}")
    print("-" * 30)

    for hidden, layers in [(64, 2), (128, 2), (256, 4), (512, 4)]:
        config = TPNConfig(hidden_size=hidden, num_layers=layers,
                           num_heads=4, head_dim=hidden // 4,
                           intermediate_size=hidden * 4)
        model = TPNModel(config)
        x = [0.1] * hidden

        t = timeit(lambda: model.forward(x), repeat=2)
        print(f"{hidden:>8}{layers:>8}{t*1000:>14.3f}")


def main():
    print()
    print("#" * 74)
    print("#  TPN PERFORMANCE BASELINE (pure Python)")
    print("#" * 74)
    bench_matvec()
    bench_attention()
    bench_hadamard()
    bench_forward()
    print()
    print("=" * 74)
    print("Recorded as the pre-optimization baseline.")
    print("=" * 74)


if __name__ == "__main__":
    main()

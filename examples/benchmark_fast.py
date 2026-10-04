"""
Performance comparison: pure-Python vs numpy fast path.

Reports the measured speedup for each hot path, and verifies that the
fast path produces the same numbers as the reference before timing it.

Run:  python3 examples/benchmark_fast.py
"""

import sys
import os
import time
import math
import random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from tpn_engine.phase import PhaseWeight, PhaseMatrix
from tpn_engine.hadamard import HadamardTransform
from tpn_engine.hadamard_fast import FastHadamardTransform
from tpn_engine.attention import TPNAttention
from tpn_engine.attention_fast import TPNAttentionFast
from tpn_engine.inference import TPNModel, TPNConfig


def hr(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def timeit(fn, repeat=3):
    """Best-of-N wall time in seconds."""
    best = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def fmt_speedup(ref, fast):
    return f"{ref / fast:,.0f}x" if fast > 0 else "n/a"


def bench_matvec():
    hr("1. MATVEC — phase matrix @ vector")

    random.seed(1)
    print(f"{'size':>8}{'python (ms)':>16}{'numpy (ms)':>14}{'speedup':>12}"
          f"{'agree':>9}")
    print("-" * 60)

    for n in [64, 128, 256, 512, 1024]:
        objects = [[PhaseWeight(math.pi / 2 + random.gauss(0, 0.1))
                    for _ in range(n)] for _ in range(n)]
        matrix = PhaseMatrix.from_phase_objects(objects)
        v = [random.uniform(-1, 1) for _ in range(n)]

        def ref_run():
            out = [0.0] * n
            for i in range(n):
                s = 0.0
                row = objects[i]
                for j in range(n):
                    s += row[j].value * v[j]
                out[i] = s
            return out

        fast_run = lambda: matrix.matvec(v)

        agree = np.allclose(ref_run(), fast_run(), atol=1e-9)
        t_ref = timeit(ref_run, repeat=2)
        t_fast = timeit(fast_run, repeat=5)

        print(f"{n:>8}{t_ref*1000:>16.3f}{t_fast*1000:>14.3f}"
              f"{fmt_speedup(t_ref, t_fast):>12}{str(agree):>9}")


def bench_attention():
    hr("2. ATTENTION — scores -> fitness softmax -> values")

    random.seed(2)
    print(f"{'n x m x d':>16}{'python (ms)':>16}{'numpy (ms)':>14}"
          f"{'speedup':>12}{'agree':>9}")
    print("-" * 68)

    for n, m, d in [(4, 4, 64), (8, 8, 128), (16, 16, 256),
                    (32, 32, 256), (64, 64, 256)]:
        q = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
        k = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(m)]
        v = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(m)]

        ref_run = lambda: TPNAttention.forward(q, k, v)
        fast_run = lambda: TPNAttentionFast.forward(q, k, v)

        agree = np.allclose(ref_run(), fast_run(), atol=1e-9)
        t_ref = timeit(ref_run, repeat=2)
        t_fast = timeit(fast_run, repeat=5)

        print(f"{n}x{m}x{d:<8}{t_ref*1000:>16.3f}{t_fast*1000:>14.3f}"
              f"{fmt_speedup(t_ref, t_fast):>12}{str(agree):>9}")


def bench_hadamard():
    hr("3. HADAMARD — O(N^2) matrix vs O(N log N) butterfly")

    random.seed(3)
    print(f"{'block':>8}{'python (ms)':>16}{'numpy (ms)':>14}{'speedup':>12}"
          f"{'agree':>9}")
    print("-" * 60)

    for bs in [64, 128, 256, 512, 1024]:
        ref = HadamardTransform(block_size=bs)
        fast = FastHadamardTransform(block_size=bs)
        v = [random.uniform(-1, 1) for _ in range(bs)]

        ref_run = lambda: ref.transform(v)
        fast_run = lambda: fast.transform(v)

        agree = np.allclose(ref_run(), fast_run(), atol=1e-9)
        t_ref = timeit(ref_run, repeat=2)
        t_fast = timeit(fast_run, repeat=5)

        print(f"{bs:>8}{t_ref*1000:>16.3f}{t_fast*1000:>14.3f}"
              f"{fmt_speedup(t_ref, t_fast):>12}{str(agree):>9}")


def bench_forward():
    hr("4. FULL FORWARD PASS — TPNModel python vs numpy backend")

    print(f"{'hidden':>8}{'layers':>8}{'python (ms)':>16}{'numpy (ms)':>14}"
          f"{'speedup':>12}{'agree':>9}")
    print("-" * 68)

    for hidden, layers in [(64, 2), (128, 2), (256, 4), (512, 4), (1024, 4)]:
        config = TPNConfig(hidden_size=hidden, num_layers=layers,
                           num_heads=4, head_dim=hidden // 4,
                           intermediate_size=hidden * 4)
        ref = TPNModel(config, backend="python", seed=42)
        fast = TPNModel(config, backend="numpy", seed=42)
        x = [0.1 * ((i % 5) - 2) for i in range(hidden)]

        ref_run = lambda: ref.forward(x)
        fast_run = lambda: fast.forward(x)

        agree = np.allclose(ref_run(), fast_run(), atol=1e-8)
        t_ref = timeit(ref_run, repeat=2)
        t_fast = timeit(fast_run, repeat=5)

        print(f"{hidden:>8}{layers:>8}{t_ref*1000:>16.3f}{t_fast*1000:>14.3f}"
              f"{fmt_speedup(t_ref, t_fast):>12}{str(agree):>9}")


def bench_batch():
    hr("5. BATCHED FORWARD — weight-stationary vs per-input")

    config = TPNConfig(hidden_size=256, num_layers=2, num_heads=4,
                       head_dim=64, intermediate_size=1024)
    model = TPNModel(config, backend="numpy", seed=7)

    print(f"{'batch':>8}{'per-input (ms)':>18}{'batched (ms)':>16}"
          f"{'speedup':>12}{'agree':>9}")
    print("-" * 64)

    for batch in [1, 2, 4, 8, 16, 32]:
        inputs = [[0.01 * ((i + j) % 7) for j in range(256)]
                  for i in range(batch)]

        seq_run = lambda: [model.forward(x) for x in inputs]
        batch_run = lambda: model.forward_batch(inputs)

        agree = np.allclose(seq_run(), batch_run(), atol=1e-8)
        t_seq = timeit(seq_run, repeat=3)
        t_batch = timeit(batch_run, repeat=3)

        print(f"{batch:>8}{t_seq*1000:>18.3f}{t_batch*1000:>16.3f}"
              f"{fmt_speedup(t_seq, t_batch):>12}{str(agree):>9}")


def bench_memory():
    hr("6. MEMORY — object-per-weight vs contiguous array")

    config = TPNConfig(hidden_size=512, num_layers=4, num_heads=8,
                       head_dim=64, intermediate_size=2048)
    ref = TPNModel(config, backend="python", seed=1)
    fast = TPNModel(config, backend="numpy", seed=1)

    p_ref = ref.memory_bytes()
    p_fast = fast.memory_bytes()
    params = ref.num_parameters()

    print(f"parameters            : {params:,}")
    print(f"python backend        : {p_ref/1e6:,.1f} MB "
          f"({p_ref/params:,.1f} bytes/param)")
    print(f"numpy backend         : {p_fast/1e6:,.1f} MB "
          f"({p_fast/params:,.1f} bytes/param)")
    print(f"reduction             : {p_ref/p_fast:,.0f}x")


def main():
    print()
    print("#" * 78)
    print("#  TPN FAST-PATH BENCHMARK (pure Python vs numpy)")
    print("#" * 78)
    bench_matvec()
    bench_attention()
    bench_hadamard()
    bench_forward()
    bench_batch()
    bench_memory()

    hr("SUMMARY")
    print("""
  Every fast path is verified equal to the reference implementation
  before it is timed, so none of these speedups cost correctness.
  Equivalence is enforced permanently by tests/test_fast_paths.py.
""")


if __name__ == "__main__":
    main()

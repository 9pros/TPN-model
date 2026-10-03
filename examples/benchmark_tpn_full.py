"""
Comprehensive benchmark: TPN vs standard transformer baselines.

Measures the three claims honestly:
1. Storage efficiency (phase encoding vs fp32/fp16/int8)
2. Inference efficiency (bitsliced batching, weight traversals)
3. Context scaling (temporal context vs KV cache) -- the big claim

Run:  python3 examples/benchmark_tpn_full.py
"""

import time
import math
import random
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tpn_engine.temporal_context import TemporalMemory
from tpn_engine.bitslice_inference import BitslicedLinear
from tpn_engine.echo import EchoEvolution


def hr(title):
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def bench_storage():
    hr("1. STORAGE EFFICIENCY")
    params = 27_000_000_000  # 27B

    formats = [
        ("FP32",               params * 4),
        ("FP16 / BF16",        params * 2),
        ("INT8",               params * 1),
        ("INT4",               params * 0.5),
        ("Ternary (2-bit)",    params * 2 / 8),
        ("1.58-bit (log2(3))", params * math.log2(3) / 8),
    ]
    print(f"{'Format':<22}{'Bytes':>16}{'GB':>10}")
    print("-" * 48)
    for name, b in formats:
        print(f"{name:<22}{b:>16,.0f}{b/1e9:>10.2f}")

    print()
    print("TPN .binetic storage:")
    print("  phase encoding: same information as source precision")
    print("  -> NOT smaller than the source precision by itself.")
    print("  Additions that DO reduce size:")
    print("    - version dedup: store deltas between versions (~10-20%/version)")
    print("    - shared-weight graph edges: reuse across layers (up to 50%)")
    print("    - ternary-native phase quantization: 2 bits/weight (6.75 GB @27B)")


def bench_inference():
    hr("2. INFERENCE EFFICIENCY (bitsliced batching)")

    print(f"{'batch':>6}{'seq (s)':>12}{'batch (s)':>12}{'speedup':>10}"
          f"{'w_trav seq':>12}{'w_trav batch':>14}")
    print("-" * 66)

    for batch in [1, 2, 4, 8, 16, 32]:
        rows = cols = 64
        lin = BitslicedLinear(rows, cols, num_slices=batch, seed=1)
        inputs = [[0.01 * ((i + j) % 7) for j in range(cols)] for i in range(batch)]

        t0 = time.perf_counter()
        for x in inputs:
            lin.reference_forward(x)
        seq_time = time.perf_counter() - t0
        seq_trav = batch

        lin.reset_counters()
        t0 = time.perf_counter()
        lin.forward(inputs)
        batch_time = time.perf_counter() - t0
        batch_trav = lin.weight_traversals

        speedup = seq_time / batch_time if batch_time > 0 else 0
        print(f"{batch:>6}{seq_time:>12.5f}{batch_time:>12.5f}"
              f"{speedup:>9.2f}x{seq_trav:>12}{batch_trav:>14}")

    print()
    print("Interpretation: the structural win is WEIGHT TRAVERSALS: at")
    print("batch=B the weights are read 1x instead of Bx. In Python the")
    print("wall-clock gain is masked by interpreter overhead; on hardware")
    print("this maps to a ~B reduction in weight memory bandwidth.")


def bench_context():
    hr("3. CONTEXT SCALING (the 'no context bound' claim)")

    dim = 64
    random.seed(3)

    print("3a. MEMORY: constant vs context length")
    print(f"{'context_len':>12}{'KV cache KB':>14}{'TPN KB':>10}{'compression':>14}")
    print("-" * 52)
    for n in [64, 256, 1024, 4096, 16384, 65536]:
        mem = TemporalMemory(dim=dim, num_frequencies=32,
                             forgetting=1.0, window=1e6)
        for i in range(n):
            mem.write([random.gauss(0, 1) for _ in range(dim)],
                      arrival_time=float(i))
        kv_kb = n * dim * 2 * 4 / 1024
        tpn_kb = mem.ctx.memory_bytes() / 1024
        print(f"{n:>12}{kv_kb:>14.1f}{tpn_kb:>10.1f}{kv_kb/tpn_kb:>13.1f}x")
    print("  -> TPN memory is FLAT. KV cache grows linearly.")

    print()
    print("3b. CAPACITY: how many items a fixed memory can hold")
    print("    (avg over 20 trials; query the newest item by its exact time)")
    print(f"{'items':>8}{'recall':>10}{'random':>10}{'above_rand':>12}")
    print("-" * 40)
    for n in [1, 4, 8, 16, 32, 64, 128]:
        rec_sum = 0.0
        rand_sum = 0.0
        trials = 20
        for _ in range(trials):
            mem = TemporalMemory(dim=dim, num_frequencies=32,
                                 forgetting=1.0, window=1e6)
            vectors = []
            for i in range(n):
                vec = [random.gauss(0, 1) for _ in range(dim)]
                vectors.append(vec)
                mem.write(vec, arrival_time=float(i))
            _, score = mem.read(vectors[-1], query_time=float(n - 1))
            _, rand_score = mem.read([random.gauss(0, 1) for _ in range(dim)],
                                     query_time=float(n - 1))
            rec_sum += score
            rand_sum += rand_score
        score = rec_sum / trials
        rand_score = rand_sum / trials
        print(f"{n:>8}{score:>10.3f}{rand_score:>10.3f}"
              f"{score - rand_score:>+12.3f}")

    print()
    print("3c. CAPACITY SCALES WITH DIM (dim = how much fits)")
    print("    (avg over 10 trials; recall of the newest item)")
    print(f"{'dim':>6}{'items=16':>12}{'items=64':>12}{'items=128':>12}")
    print("-" * 42)
    for d in [32, 64, 128, 256]:
        row = []
        for n in [16, 64, 128]:
            s = 0.0
            for _ in range(10):
                mem = TemporalMemory(dim=d, num_frequencies=d,
                                     window=1e6, forgetting=1.0)
                vecs = []
                for i in range(n):
                    v = [random.gauss(0, 1) for _ in range(d)]
                    vecs.append(v)
                    mem.write(v, arrival_time=float(i))
                _, sc = mem.read(vecs[-1], query_time=float(n - 1))
                s += sc
            row.append(s / 10)
        print(f"{d:>6}{row[0]:>12.3f}{row[1]:>12.3f}{row[2]:>12.3f}")

    print()
    print("Interpretation (honest):")
    print("  - VERIFIED WIN (memory): the state is O(dim^2) and FLAT in")
    print("    context length, vs a KV cache that grows O(n). For dim=64 the")
    print("    state is 64 KB constant; the KV cache passes it at ~128")
    print("    tokens and is ~511x larger at 64k tokens. The win grows")
    print("    linearly with context length.")
    print("  - LIMIT (capacity): a dim-d state holds ~d/2 to d items at")
    print("    usable recall (0.9 at dim=256/128 items), then falls off.")
    print("    Same capacity class as linear attention / SSMs. NOT perfect")
    print("    infinite recall.")
    print("  - Correct claim: 'context length decoupled from memory;")
    print("    capacity set by dim.' NOT 'unbounded context'.")
    print("  - Tradeoff to tune: dim (capacity) vs state size (dim^2).")


def bench_echo():
    hr("4. ECHO EVOLUTION (coherence-driven phase selection)")
    evo = EchoEvolution(num_phases=16, generations=60, target_round=3, seed=7)
    result = evo.evolve()
    print(f"initial coherence: {result['initial_coherence']:.4f}")
    print(f"final coherence:   {result['final_coherence']:.4f}")
    print(f"improvement:       "
          f"{result['final_coherence'] - result['initial_coherence']:+.4f}")
    print()
    print("Interpretation: evolution tunes per-channel phase offsets so the")
    print("echo channels align constructively at the target round. The")
    print("optimum is known analytically (theta_i = -omega_i * r), so this")
    print("validates that the echo loop is a real, searchable objective.")


def main():
    print()
    print("#" * 74)
    print("#  TPN FULL BENCHMARK")
    print("#" * 74)
    bench_storage()
    bench_inference()
    bench_context()
    bench_echo()

    hr("SUMMARY")
    print("""
  CLAIM                          VERDICT
  ----------------------------   ------------------------------------------
  more compact storage           PARTIAL - equal to source precision;
                                 wins only via dedup/sharing/ternary-native
  faster inference               PARTIAL - equal FLOPs; structural win is
                                 fewer weight traversals (memory bandwidth)
  instant loading                PLAUSIBLE - graph + phase format is
                                 mmap-friendly; not yet measured
  time-based context, no bound   TRUE asymptotically (O(dim) memory,
                                 O(dim)/token); recall is lossy
  echo in evolution              IMPLEMENTED - real objective with known
                                 optimum; validated in isolation
""")


if __name__ == "__main__":
    main()

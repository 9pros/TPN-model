"""
TPN Benchmark - Compare TPN vs Standard attention.
"""

import time
import math
import random
from tpn_engine import TPNAttention, tpn_softmax, standard_softmax


def benchmark_attention():
    print("=" * 60)
    print("TPN vs Standard Attention Benchmark")
    print("=" * 60)
    print()
    
    sizes = [
        (4, 4, 64, "Small"),
        (8, 8, 128, "Medium"),
        (16, 16, 256, "Large"),
    ]
    
    for n, m, d, label in sizes:
        print(f"{label} ({n}x{m}x{d}):")
        
        # Generate random matrices
        random.seed(42)
        q = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(n)]
        k = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(m)]
        v = [[random.uniform(-1, 1) for _ in range(d)] for _ in range(m)]
        
        # Benchmark TPN attention
        start = time.time()
        for _ in range(100):
            output_tpn = TPNAttention.forward(q, k, v)
        tpn_time = (time.time() - start) / 100
        
        # Benchmark standard attention
        start = time.time()
        for _ in range(100):
            output_std = TPNAttention.forward_standard(q, k, v)
        std_time = (time.time() - start) / 100
        
        print(f"  TPN:      {tpn_time*1000:.3f} ms")
        print(f"  Standard: {std_time*1000:.3f} ms")
        print(f"  Speedup:  {std_time/tpn_time:.2f}x")
        print()


if __name__ == "__main__":
    benchmark_attention()

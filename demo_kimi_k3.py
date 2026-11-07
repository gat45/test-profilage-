#!/usr/bin/env python3
"""Kimi K3 MoE optimization demo.

Simulates Kimi K3 (dense BF16 trunk + 128 MXFP4 experts, Top-16 routing) across
the three hardware platforms from Section 6 of the architecture document:
  1. Ryzen 5 5600 + RTX 3060 12GB
  2. Ryzen AI 9 HX365 + RTX 5070 8GB
  3. OnePlus 15 (Snapdragon 8 Elite / HTP v81 / Adreno 830)
"""

import sys
sys.path.insert(0, "src")

from adaptive_quant import (
    ModelMap, CapabilityDB, MemoryProfiler, V5Predictor,
    ExpertProfiler, ExpertCacheSimulator,
)


def main():
    m = ModelMap.kimi_k3()
    db = CapabilityDB()
    mem = MemoryProfiler()
    v5 = V5Predictor(db)

    print("=" * 80)
    print("  KIMI K3 — ADAPTIVE QUANTIZATION & EXECUTION PLAN")
    print("=" * 80)
    print(m.summary())
    print()

    platforms = [
        ("ryzen5_5600_32gb", 32.0, "Platform 1 — Ryzen 5 5600 + RTX 3060 12GB"),
        ("ryzen_ai9_hx365_32gb", 32.0, "Platform 2 — Ryzen AI 9 HX365 + RTX 5070 8GB"),
        ("oneplus_15_htp_v81", 16.0, "Platform 3 — OnePlus 15 (HTP v81 + Adreno 830)"),
    ]

    for dev_name, mem_b, label in platforms:
        dev = db.get(dev_name)
        tier = "ufs4_128gb" if dev.device_type == "soc" else "nvme_512gb"
        print("-" * 80)
        print(f"  {label}")
        print("-" * 80)
        r = v5.predict(m, dev, storage_tier=tier, memory_budget_gb=mem_b, context_tokens=4096)
        print(v5.print(r))
        print()

    # MoE cache optimization study
    print("=" * 80)
    print("  MOE EXPERT CACHE STUDY (128 experts, top-16, 300 tokens, concentrated routing)")
    print("=" * 80)
    ep = ExpertProfiler(128, 16)
    trace = ep.synthetic_trace(n_tokens=300, distribution="concentrated", seed=42)
    stats = ep.analyze(trace)
    print(ep.print_report(stats))
    print()
    sim = ExpertCacheSimulator(bytes_per_expert=300 * 1024 * 1024, storage_bw_mbps=4000.0)
    res = sim.compare_policies(trace, 32)
    print(ExpertCacheSimulator.compare_table(res))
    opt = sim.optimize_size(trace, target_hit_rate=0.85, max_experts=128, policy="adaptive")
    print(f"\n→ Adaptive cache size for 85% hit rate: {opt.capacity_experts} experts "
          f"({opt.cache_size_mb:.0f} MB), hit rate {opt.hit_rate*100:.1f}%, T_cache {opt.cache_time_ms:.0f} ms")


if __name__ == "__main__":
    main()

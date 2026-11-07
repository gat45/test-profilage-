#!/usr/bin/env python3
"""GTX 1080 validation reproduction.

Reproduces the documented Section 5.1 results on Qwen3.5-9B (mixed vs uniform Q4_0),
runs the planner and validation engine, and compares predicted vs measured values.
"""

import sys
sys.path.insert(0, "src")

from adaptive_quant import (
    ModelMap, CapabilityDB, MixedPrecisionSolver, CostModel, MemoryProfiler,
    Z2D2Planner, ValidationEngine,
)


def main():
    m = ModelMap.qwen35_9b()
    db = CapabilityDB()
    dev = db.get("gtx_1080_8gb")
    slv = MixedPrecisionSolver()
    cost = CostModel()
    mem = MemoryProfiler()

    print("=" * 80)
    print("  GTX 1080 VALIDATION — QWEN3.5-9B (Section 5.1 reproduction)")
    print("=" * 80)

    budget = int(6.6 * (1024 ** 3))          # storage budget: leaves KV/act room in 8GB VRAM
    mixed = slv.solve_greedy(m, budget, name="mixed")
    q4 = slv.uniform_plan(m, "Q4_0")
    q8 = slv.uniform_plan(m, "Q8_0")

    print(slv.compare([mixed, q4, q8]))
    print()

    # Costs
    c_m = cost.t_token(m, dev, mixed)
    c_q = cost.t_token(m, dev, q4)
    print(cost.print_breakdown(m.name, dev.name, c_m))
    print()

    # Memory
    mp = mem.profile(m, dev, quant_plan=mixed.plan)
    print(mem.print_budget_table(mp))
    print()

    # Comparison table
    print("-" * 80)
    print("  MEASURED vs PREDICTED")
    print("-" * 80)
    print(f"{'Metric':<14} {'Documented (mixed)':<24} {'Predicted (mixed)':<20} {'Measured vs Pred'}")
    print("-" * 80)
    print(f"{'decode tok/s':<14} {'33.42':<24} {str(c_m.decode_tok_s):<20} {c_m.decode_tok_s - 33.42:+.1f} tok/s")
    print(f"{'prefill tok/s':<14} {'~35-40':<24} {str(c_m.prefill_tok_s):<20}")
    print(f"{'PPL (est.)':<14} {'9.7839':<24} {'Δ ~ +0.15 vs FP16':<20}")
    print(f"{'Q4 decode':<14} {'35.98':<24} {str(c_q.decode_tok_s):<20} {c_q.decode_tok_s - 35.98:+.1f} tok/s")
    print(f"{'GPU size':<14} {'5.388 GiB (mixed)':<24} {f'{mixed.total_size_gb:.3f} GB':<20}")
    print()

    # Validation
    print("=" * 80)
    print("  L1–L3 VALIDATION OF THE MIXED PLAN")
    print("=" * 80)
    rep = ValidationEngine().validate_plan(m, mixed, mem=mp, cost=c_m, device=dev)
    print(ValidationEngine.print(rep))


if __name__ == "__main__":
    main()

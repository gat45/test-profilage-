"""Unified CLI for the adaptive quantization and hardware profiling framework.

Commands:
  plan            — end-to-end Z2/D2 allocation for a model+device
  profile         — structural and memory profiling
  predict         — offline performance prediction (or --v5 advanced)
  simulate-cache  — MoE expert cache simulation
  validate        — L1–L3 validation of a plan
  export-map      — write model_map.json / layer_map.csv / tensor_map.csv
  parse-htp       — analyze Qualcomm HTP trace (VTCM, CPU fallbacks)
  devices         — list known hardware profiles
  formats         — list known numerical formats
  demo-gtx1080    — reproduce the documented GTX 1080 validation
  demo-kimi       — simulate Kimi K3 on the 3 target hardware platforms
"""

from __future__ import annotations
import argparse
import sys
from .capability_db import CapabilityDB
from .quant_formats import quant_registry
from .profile_model import ModelMap
from .mixed_precision import MixedPrecisionSolver
from .profiler import MemoryProfiler
from .cost_model import CostModel
from .predict_from_hf import OfflinePredictor
from .predict_from_hf_v5 import V5Predictor
from .expert_profiler import ExpertProfiler
from .expert_cache import ExpertCacheSimulator
from .parse_hexagon import HexagonTraceParser
from .planner import Z2D2Planner
from .validation import ValidationEngine


MODEL_PRESETS = {
    "qwen35_9b": ModelMap.qwen35_9b,
    "bonsai_27b": ModelMap.bonsai_27b,
    "kimi_k3": ModelMap.kimi_k3,
    "marco_moe": ModelMap.marco_moe,
}


def _get_model(name: str) -> ModelMap:
    fn = MODEL_PRESETS.get(name.lower())
    if fn is None:
        raise SystemExit(f"Unknown preset '{name}'. Presets: {list(MODEL_PRESETS.keys())}")
    return fn()


def cmd_plan(a: argparse.Namespace) -> None:
    m = _get_model(a.model)
    db = CapabilityDB()
    p = Z2D2Planner(db).run(m, a.device, memory_budget_gb=a.budget, context_tokens=a.ctx,
                            target_format=a.fmt, solver_kind=a.solver)
    print(p.print())
    if a.out:
        p.export(a.out)
        print(f"\nExported quantization_plan.json, execution_strategy.json, decision_log.json -> {a.out}/")


def cmd_profile(a: argparse.Namespace) -> None:
    m = _get_model(a.model)
    print(m.summary())
    db = CapabilityDB()
    dev = db.get(a.device)
    mp = MemoryProfiler().profile(m, dev, context_tokens=a.ctx,
                                  kv_precision="int4" if dev.total_addressable_gb < 10 else "fp16",
                                  memory_budget_gb=a.budget or dev.total_addressable_gb)
    print()
    print(MemoryProfiler().print_budget_table(mp))


def cmd_predict(a: argparse.Namespace) -> None:
    m = _get_model(a.model)
    db = CapabilityDB()
    dev = db.get(a.device)
    if a.v5:
        r = V5Predictor(db).predict(m, dev, memory_budget_gb=a.budget, context_tokens=a.ctx)
        print(V5Predictor().print(r))
    else:
        r = OfflinePredictor(db).predict(m, dev, memory_budget_gb=a.budget, context_tokens=a.ctx,
                                         target_format=a.fmt)
        print(OfflinePredictor().print(r))


def cmd_sim_cache(a: argparse.Namespace) -> None:
    ep = ExpertProfiler(a.experts, a.topk)
    trace = ep.synthetic_trace(n_tokens=a.tokens, distribution=a.dist)
    stats = ep.analyze(trace)
    print(ep.print_report(stats))
    print()
    sim = ExpertCacheSimulator(bytes_per_expert=a.expert_mb * 1024 * 1024, storage_bw_mbps=a.bw)
    res = sim.compare_policies(trace, a.cap)
    print(ExpertCacheSimulator.compare_table(res))
    opt = sim.optimize_size(trace, target_hit_rate=a.target_hr, max_experts=a.experts, policy="adaptive")
    print(f"\nAdaptive cache size for {a.target_hr*100:.0f}% hit rate: {opt.capacity_experts} experts "
          f"({opt.cache_size_mb:.0f} MB), hit rate {opt.hit_rate*100:.1f}%")


def cmd_validate(a: argparse.Namespace) -> None:
    m = _get_model(a.model)
    db = CapabilityDB()
    dev = db.get(a.device)
    p = Z2D2Planner(db).run(m, a.device, memory_budget_gb=a.budget)
    rep = ValidationEngine().validate_plan(m, p.allocation_plan, p.memory_profile, p.cost_breakdown, dev)
    print(ValidationEngine().print(rep))


def cmd_export_map(a: argparse.Namespace) -> None:
    m = _get_model(a.model)
    m.export(a.out)
    print(f"Exported model_map.json, layer_map.csv, tensor_map.csv, architecture_report.json -> {a.out}/")


def cmd_parse_htp(a: argparse.Namespace) -> None:
    txt = HexagonTraceParser.bonsai_stress_trace() if a.trace == "bonsai-stress" else (
        HexagonTraceParser.healthy_trace() if a.trace == "healthy" else open(a.trace).read()
    )
    t = HexagonTraceParser().parse(txt, source=a.trace)
    print(HexagonTraceParser.print(t))


def cmd_devices(a: argparse.Namespace) -> None:
    for d in CapabilityDB().list_devices():
        print(f"{d.name:<24} | {d.label:<60} | {d.device_type} | {d.total_addressable_gb} GB")


def cmd_formats(a: argparse.Namespace) -> None:
    for f in quant_registry:
        print(f"{f.name:<10} {f.bpw():>7.3f} bpw  block={f.block_size:<4} {f.category:<10}  {f.label}")


def cmd_demo_gtx(a: argparse.Namespace) -> None:
    m = ModelMap.qwen35_9b()
    db = CapabilityDB()
    dev = db.get("gtx_1080_8gb")
    slv = MixedPrecisionSolver()
    budget = int(8.0 * (1024 ** 3))
    mixed = slv.solve_greedy(m, budget)
    q4 = slv.uniform_plan(m, "Q4_0")
    print("=== GTX 1080 validation reproduction (Qwen3.5-9B) ===")
    print(slv.compare([mixed, q4]))
    cost = CostModel()
    c_m = cost.t_token(m, dev, mixed)
    c_q = cost.t_token(m, dev, q4)
    print()
    print(f"Predicted decode : mixed={c_m.decode_tok_s} tok/s  |  Q4_0={c_q.decode_tok_s} tok/s")
    print(f"Documented meas. : mixed=33.42 tok/s / PPL 9.7839  |  Q4=35.98 tok/s / PPL 10.0437")
    print("Mixed precision improves quality (+2.6% PPL) at a small throughput cost.")


def cmd_demo_kimi(a: argparse.Namespace) -> None:
    m = ModelMap.kimi_k3()
    db = CapabilityDB()
    platforms = [("ryzen5_5600_32gb", 32.0, "Simulation platform 1: Ryzen 5 5600 + RTX 3060"),
                 ("ryzen_ai9_hx365_32gb", 32.0, "Simulation platform 2: Ryzen AI 9 HX365"),
                 ("oneplus_15_htp_v81", 16.0, "Simulation platform 3: OnePlus 15 (HTP v81)")]
    print(f"=== Kimi K3 MoE Simulation ({m.total_params/1e9:.0f}B params, 128 MXFP4 experts, Top-16) ===")
    print(f"    dense trunk BF16 + MXFP4 experts; active per token: {m.n_params_per_token_decode/1e9:.1f}B params")
    print()
    v5 = V5Predictor(db)
    for dev_name, mem_b, label in platforms:
        dev = db.get(dev_name)
        tier = "ufs4_128gb" if dev.device_type == "soc" else "nvme_512gb"
        r = v5.predict(m, dev, storage_tier=tier, memory_budget_gb=mem_b, context_tokens=4096)
        print(f"--- {label} ---")
        print(v5.print(r))
        print()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="quant", description="Adaptive quantization & profiling CLI")
    sub = p.add_subparsers(dest="cmd")

    sp = sub.add_parser("plan"); sp.add_argument("--model", default="kimi_k3"); sp.add_argument("--device", default="oneplus_15_htp_v81"); sp.add_argument("--budget", type=float); sp.add_argument("--ctx", type=int, default=4096); sp.add_argument("--fmt", default="auto"); sp.add_argument("--solver", default="greedy"); sp.add_argument("--out")
    sp.set_defaults(func=cmd_plan)

    sp = sub.add_parser("profile"); sp.add_argument("--model", default="kimi_k3"); sp.add_argument("--device", default="oneplus_15_htp_v81"); sp.add_argument("--budget", type=float); sp.add_argument("--ctx", type=int, default=4096)
    sp.set_defaults(func=cmd_profile)

    sp = sub.add_parser("predict"); sp.add_argument("--model", default="kimi_k3"); sp.add_argument("--device", default="oneplus_15_htp_v81"); sp.add_argument("--budget", type=float); sp.add_argument("--ctx", type=int, default=4096); sp.add_argument("--fmt", default="auto"); sp.add_argument("--v5", action="store_true")
    sp.set_defaults(func=cmd_predict)

    sp = sub.add_parser("simulate-cache"); sp.add_argument("--experts", type=int, default=128); sp.add_argument("--topk", type=int, default=16); sp.add_argument("--tokens", type=int, default=200); sp.add_argument("--cap", type=int, default=16); sp.add_argument("--expert-mb", type=float, default=250.0); sp.add_argument("--bw", type=float, default=4000.0); sp.add_argument("--dist", default="concentrated"); sp.add_argument("--target-hr", type=float, default=0.85)
    sp.set_defaults(func=cmd_sim_cache)

    sp = sub.add_parser("validate"); sp.add_argument("--model", default="qwen35_9b"); sp.add_argument("--device", default="gtx_1080_8gb"); sp.add_argument("--budget", type=float, default=8.0)
    sp.set_defaults(func=cmd_validate)

    sp = sub.add_parser("export-map"); sp.add_argument("--model", default="kimi_k3"); sp.add_argument("--out", default="out")
    sp.set_defaults(func=cmd_export_map)

    sp = sub.add_parser("parse-htp"); sp.add_argument("--trace", default="bonsai-stress")
    sp.set_defaults(func=cmd_parse_htp)

    sp = sub.add_parser("devices"); sp.set_defaults(func=cmd_devices)
    sp = sub.add_parser("formats"); sp.set_defaults(func=cmd_formats)
    sp = sub.add_parser("demo-gtx1080"); sp.set_defaults(func=cmd_demo_gtx)
    sp = sub.add_parser("demo-kimi"); sp.set_defaults(func=cmd_demo_kimi)
    return p


def main(argv=None) -> None:
    p = build_parser()
    a = p.parse_args(argv)
    if not a.cmd:
        p.print_help()
        return
    a.func(a)


if __name__ == "__main__":
    main()

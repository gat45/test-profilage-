"""Advanced V5 prediction (predict_from_hf_v5.py — Profiler Axe 12 V5 extension).

V5 adds: storage ceilings per storage tier, calibrated backend confidence
intervals, MTP speculative decoding projection, and a Pareto frontier of
quantization candidates for the target model+device.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from .profile_model import ModelMap
from .capability_db import CapabilityDB, DeviceProfile
from .mixed_precision import MixedPrecisionSolver, AllocationPlan
from .profiler import MemoryProfiler
from .cost_model import CostModel
from .predict_from_hf import PredictionReport


STORAGE_CEILINGS_GB = {
    "ufs4_128gb": 128.0,
    "ufs4_256gb": 256.0,
    "nvme_512gb": 512.0,
    "nvme_1tb": 1024.0,
    "nvme_2tb": 2048.0,
}

# Empirical validation anchors from the document
VALIDATION_ANCHORS = {
    ("gtx_1080_8gb", "Qwen3.5-9B", "mixed"): 33.42,
    ("gtx_1080_8gb", "Qwen3.5-9B", "Q4_0"): 35.98,
    ("gtx_1080_ti_11gb", "Bonsai-27B", "Q1_0"): 28.3,
    ("oneplus_15_htp_v81", "Marco-MoE", "adreno_opencl"): 31.23,
    ("oneplus_15_htp_v81", "Marco-MoE", "htp"): 23.6,
    ("oneplus_15_htp_v81", "1.7B", "adreno_opencl"): 38.6,
    ("oneplus_15_htp_v81", "1.7B", "htp"): 32.5,
    ("oneplus_15_htp_v81", "9B", "adreno_opencl"): 10.19,
    ("oneplus_15_htp_v81", "9B", "htp"): 9.8,
}


@dataclass
class V5Report:
    model_name: str
    device_name: str
    # 3 budgets
    storage_budget_gb: float
    memory_budget_gb: float
    movement_gb_per_1000tokens: float
    storage_ceiling_gb: float
    storage_headroom_gb: float
    storage_ok: bool
    # selected plan
    selected_plan: str
    average_bpw: float
    working_set_gb: float
    decode_tok_s: float
    decode_tok_s_lo: float
    decode_tok_s_hi: float
    decode_with_mtp_tok_s: float      # MTP speculation amplification
    saturation_risk: str
    io_dominated: bool
    calibrated_against_anchor: Optional[float]
    pareto_plans: List[dict]
    notes: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


class V5Predictor:
    """Advanced predictor with ceilings, calibrated uncertainty, MTP projection."""

    def __init__(self, db: Optional[CapabilityDB] = None) -> None:
        self.db = db or CapabilityDB()
        self.solver = MixedPrecisionSolver()
        self.profiler = MemoryProfiler()
        self.cost = CostModel()

    def predict(
        self,
        model: ModelMap,
        device: DeviceProfile,
        storage_tier: str = "nvme_512gb",
        memory_budget_gb: Optional[float] = None,
        context_tokens: int = 4096,
        hit_rate: float = 0.9,
        mtp_gain: float = 1.7,           # doc: GTX 1080 MTP 55-61 tok/s vs ~33 baseline ≈ 1.65-1.8x
    ) -> V5Report:
        budget_gb = memory_budget_gb or device.total_addressable_gb or 8.0
        budget_bytes = int(budget_gb * (1024 ** 3))
        ceiling_gb = STORAGE_CEILINGS_GB.get(storage_tier, 512.0)
        notes: List[str] = []

        # Plans
        best = self.solver.solve_greedy(model, budget_bytes)
        mem = self.profiler.profile(
            model, device, quant_plan=best.plan, context_tokens=context_tokens,
            kv_precision="int4" if budget_gb < 10 else "fp16",
            memory_budget_gb=budget_gb,
        )
        cost = self.cost.t_token(model, device, plan=best, hit_rate=hit_rate)

        # Storage ceiling check
        stor_gb = mem.storage_budget_gb
        headroom = ceiling_gb - stor_gb
        if headroom < 20:
            notes.append(f"Storage ceiling exceeded/near: {stor_gb:.1f} GB of {ceiling_gb:.0f} GB ({storage_tier}).")

        # Calibrated intervals
        unc = 0.10 if device.name in ("gtx_1080_8gb", "gtx_1080_ti_11gb") else (0.25 if device.validated else 0.40)
        lo = round(cost.decode_tok_s * (1 - unc), 1)
        hi = round(cost.decode_tok_s * (1 + unc), 1)
        anchor = VALIDATION_ANCHORS.get((device.name, model.name, "mixed")) or VALIDATION_ANCHORS.get((device.name, model.name, "Q4_0"))

        risk = "high" if mem.exceeds_budget else ("medium" if mem.working_set_gb > budget_gb * 0.85 else "low")
        mtp = round(cost.decode_tok_s * mtp_gain, 1) if cost.decode_tok_s > 0 else 0.0

        pareto = self.solver.solve_pareto(model, n_points=6, min_budget_gb=budget_gb * 0.5,
                                           max_budget_gb=budget_gb * 1.5)
        pareto_rows = [{"plan": p.name, "size_gb": round(p.total_size_gb, 2),
                        "avg_bpw": round(p.average_bpw, 2),
                        "weighted_error": p.total_weighted_error} for p in pareto]

        return V5Report(
            model_name=model.name,
            device_name=device.name,
            storage_budget_gb=mem.storage_budget_gb,
            memory_budget_gb=mem.memory_budget_gb,
            movement_gb_per_1000tokens=round(mem.cold_bytes_per_token * 1000 / (1024 ** 3), 3),
            storage_ceiling_gb=ceiling_gb,
            storage_headroom_gb=round(headroom, 1),
            storage_ok=headroom >= 20,
            selected_plan=best.name,
            average_bpw=round(best.average_bpw, 2),
            working_set_gb=mem.working_set_gb,
            decode_tok_s=cost.decode_tok_s,
            decode_tok_s_lo=lo,
            decode_tok_s_hi=hi,
            decode_with_mtp_tok_s=mtp,
            saturation_risk=risk,
            io_dominated=cost.io_dominated,
            calibrated_against_anchor=anchor,
            pareto_plans=pareto_rows,
            notes=notes + ([cost.notes] if cost.notes else []),
        )

    def print(self, r: V5Report) -> str:
        lines = [
            f"=== V5 Advanced Prediction : {r.model_name} on {r.device_name} ===",
            f"  3 Budgets : A={r.storage_budget_gb:.1f} GB storage | B={r.memory_budget_gb:.1f} GB memory | C={r.movement_gb_per_1000tokens:.3f} GB/1k-tokens",
            f"  storage ceiling ({r.storage_ceiling_gb:.0f} GB tier) : {'OK' if r.storage_ok else 'EXCEEDS'} ({r.storage_headroom_gb:.0f} GB free)",
            f"  working set  : {r.working_set_gb} GB → risk: {r.saturation_risk}",
            f"  decode       : {r.decode_tok_s} tok/s  [{r.decode_tok_s_lo}, {r.decode_tok_s_hi}]",
            f"  with MTP×1.7 : {r.decode_with_mtp_tok_s} tok/s  | bound: {'I/O' if r.io_dominated else 'compute'}",
        ]
        if r.calibrated_against_anchor:
            lines.append(f"  validation anchor (measured) : {r.calibrated_against_anchor} tok/s")
        if r.notes:
            lines.append("  notes: " + " | ".join(r.notes))
        lines.append("  Pareto frontier:")
        for p in r.pareto_plans:
            lines.append(f"    {p['plan']:<10} {p['size_gb']:>6.1f} GB  {p['avg_bpw']:>5.2f} bpw  err={p['weighted_error']:.3e}")
        return "\n".join(lines)

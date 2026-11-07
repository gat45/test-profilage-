"""Offline performance prediction (Profiler Axe 12 — Performance prediction).

Predicts final size, working set, peak memory, cold bytes/token, dequant time,
compute time, throughput and saturation risk BEFORE running any quantization.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, Optional
from .profile_model import ModelMap
from .capability_db import CapabilityDB, DeviceProfile
from .mixed_precision import MixedPrecisionSolver
from .profiler import MemoryProfiler
from .cost_model import CostModel


@dataclass
class PredictionReport:
    model_name: str
    device_name: str
    plan_name: str
    total_params_B: float
    storage_size_gb: float
    average_bpw: float
    resident_gb: float
    kv_cache_gb: float
    working_set_gb: float
    peak_rss_gb: float
    memory_budget_gb: float
    budget_ok: bool
    cold_bytes_per_tok_mb: float
    io_dominated: bool
    t_dequant_ms: float
    t_compute_ms: float
    t_io_ms: float
    decode_tok_s: float
    prefill_tok_s: float
    decode_range: str                 # "[lo, hi]" confidence interval
    saturation_risk: str              # "low", "medium", "high"
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class OfflinePredictor:
    """Combines model profiling, solver planning, and cost modeling into one prediction pass."""

    def __init__(self, db: Optional[CapabilityDB] = None) -> None:
        self.db = db or CapabilityDB()
        self.solver = MixedPrecisionSolver()
        self.profiler = MemoryProfiler()
        self.cost = CostModel()

    def predict(
        self,
        model: ModelMap,
        device: DeviceProfile,
        memory_budget_gb: Optional[float] = None,
        context_tokens: int = 4096,
        target_format: str = "auto",       # "auto" (mixed), or specific format name
        hit_rate: float = 0.9,
    ) -> PredictionReport:
        budget_gb = memory_budget_gb or device.total_addressable_gb or 8.0
        budget_bytes = int(budget_gb * (1024 ** 3))

        # (1) build plan
        if target_format == "auto":
            plan = self.solver.solve_greedy(model, budget_bytes)
        else:
            plan = self.solver.uniform_plan(model, target_format)

        # (2) memory profile
        mem = self.profiler.profile(
            model, device, quant_plan=plan.plan, context_tokens=context_tokens,
            kv_precision="int4" if budget_gb < 10 else "fp16",
            memory_budget_gb=budget_gb,
        )

        # (3) cost breakdown
        cost = self.cost.t_token(model, device, plan=plan, hit_rate=hit_rate)

        # (4) saturation risk
        if mem.exceeds_budget:
            risk = "high"
        elif mem.working_set_gb > budget_gb * 0.85:
            risk = "medium"
        else:
            risk = "low"

        # (5) confidence interval (calibrated vs uncalibrated device)
        uncertainty = 0.10 if device.validated else 0.35
        lo = round(cost.decode_tok_s * (1 - uncertainty), 1)
        hi = round(cost.decode_tok_s * (1 + uncertainty), 1)

        return PredictionReport(
            model_name=model.name,
            device_name=device.name,
            plan_name=plan.name,
            total_params_B=round(model.total_params / 1e9, 2),
            storage_size_gb=round(plan.total_size_gb, 2),
            average_bpw=round(plan.average_bpw, 2),
            resident_gb=mem.resident_weights_gb,
            kv_cache_gb=mem.kv_cache_gb,
            working_set_gb=mem.working_set_gb,
            peak_rss_gb=mem.peak_rss_gb,
            memory_budget_gb=budget_gb,
            budget_ok=not mem.exceeds_budget,
            cold_bytes_per_tok_mb=round(mem.cold_bytes_per_token / (1024 ** 2), 2),
            io_dominated=cost.io_dominated,
            t_dequant_ms=cost.t_dequant_ms,
            t_compute_ms=cost.t_compute_ms,
            t_io_ms=cost.t_io_ms,
            decode_tok_s=cost.decode_tok_s,
            prefill_tok_s=cost.prefill_tok_s,
            decode_range=f"[{lo}, {hi}]",
            saturation_risk=risk,
            notes=cost.notes,
        )

    def print(self, r: PredictionReport) -> str:
        return (
            f"=== Prediction : {r.model_name} on {r.device_name} ===\n"
            f"  params: {r.total_params_B}B  |  plan: {r.plan_name} ({r.average_bpw} bpw, {r.storage_size_gb} GB on disk)\n"
            f"  working set: {r.working_set_gb} GB / {r.memory_budget_gb} GB budget "
            f"({'OK' if r.budget_ok else 'EXCEEDS'})  |  risk: {r.saturation_risk}\n"
            f"    resident={r.resident_gb} GB | KV={r.kv_cache_gb} GB | cold IO={r.cold_bytes_per_tok_mb} MB/tok "
            f"({'I/O bound' if r.io_dominated else 'compute bound'})\n"
            f"  decode: {r.decode_tok_s} tok/s (range {r.decode_range})  |  prefill: {r.prefill_tok_s} tok/s\n"
            f"  breakdown: IO={r.t_io_ms:.0f}ms  compute={r.t_compute_ms:.0f}ms  dequant={r.t_dequant_ms:.0f}ms"
            + (f"\n  notes: {r.notes}" if r.notes else "")
        )


def predict(
    model: ModelMap,
    device_name: str,
    memory_budget_gb: Optional[float] = None,
    context_tokens: int = 4096,
    target_format: str = "auto",
) -> PredictionReport:
    """One-line convenience prediction."""
    db = CapabilityDB()
    device = db.get(device_name)
    return OfflinePredictor(db).predict(model, device, memory_budget_gb, context_tokens, target_format)

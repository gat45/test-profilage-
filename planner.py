"""Z2/D2 Planner — Decision and resource allocation pipeline (section 4).

Implements the 9-step decision pipeline:
  1. Capability discovery   2. Cost model           3. Candidate generation
  4. Static Oracle          5. Prediction           6. Allocation
  7. Execution planning     8. Runtime feedback     9. Adaptive revision

Every decision is recorded with its justification for reproducibility.
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from .profile_model import ModelMap
from .capability_db import CapabilityDB, DeviceProfile
from .mixed_precision import MixedPrecisionSolver, AllocationPlan
from .profiler import MemoryProfiler, MemoryProfile
from .cost_model import CostModel, TokenCostBreakdown
from .predict_from_hf import OfflinePredictor


@dataclass
class PlanDecision:
    step: int
    name: str
    decision: str
    justification: str


@dataclass
class ExecutionStrategy:
    resident_tensors: List[str]
    streaming_tensors: List[str]
    kv_precision: str
    expert_cache_capacity_experts: int
    backend_placements: Dict[str, str]   # "cpu" / "gpu" / "htp" per family
    streaming_enabled: bool
    estimated_decode_tok_s: float
    estimated_working_set_gb: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PlannerResult:
    model_name: str
    device_name: str
    allocation_plan: AllocationPlan
    memory_profile: MemoryProfile
    cost_breakdown: TokenCostBreakdown
    execution_strategy: ExecutionStrategy
    decisions: List[PlanDecision]
    measured_feedback: Optional[dict] = None
    adapted: bool = False

    def export(self, out_dir: str = ".") -> None:
        import os
        os.makedirs(out_dir, exist_ok=True)
        self.allocation_plan.export(f"{out_dir}/quantization_plan.json")
        with open(f"{out_dir}/execution_strategy.json", "w") as f:
            json.dump(self.execution_strategy.to_dict(), f, indent=2)
        with open(f"{out_dir}/decision_log.json", "w") as f:
            json.dump([asdict(d) for d in self.decisions], f, indent=2)

    def print(self) -> str:
        s = self.allocation_plan
        mp = self.memory_profile
        cb = self.cost_breakdown
        es = self.execution_strategy
        lines = [
            f"=== Z2/D2 Plan : {self.model_name} on {self.device_name} ===",
            f"  [1] solver: {s.solver} | budget: {mp.memory_budget_gb} GB | storage: {s.total_size_gb:.1f} GB ({s.average_bpw:.2f} bpw)",
            f"  [2] working set: {mp.working_set_gb:.1f} GB ({'OK' if not mp.exceeds_budget else 'EXCEEDS'}) | resident: {mp.resident_weights_gb:.2f} GB | KV: {mp.kv_cache_gb:.2f} GB",
            f"  [3] predicted decode: {cb.decode_tok_s} tok/s | cold IO: {mp.io_per_token_mb:.2f} MB/tok | bound: {'I/O' if cb.io_dominated else 'compute'}",
            f"  [4] execution: {len(es.resident_tensors)} resident / {len(es.streaming_tensors)} streamed tensors | expert cache cap={es.expert_cache_capacity_experts}",
            "  Decisions:",
        ]
        for d in self.decisions:
            lines.append(f"    {d.step}. {d.name}: {d.decision} — {d.justification}")
        return "\n".join(lines)


class Z2D2Planner:
    """9-step orchestrator of Profiler + Quant Algo results into execution decisions."""

    def __init__(self, db: Optional[CapabilityDB] = None) -> None:
        self.db = db or CapabilityDB()
        self.solver = MixedPrecisionSolver()
        self.profiler = MemoryProfiler()
        self.cost = CostModel()
        self.last_result: Optional[PlannerResult] = None
        self._last_model: Optional[ModelMap] = None

    def run(
        self,
        model: ModelMap,
        device_name: str,
        memory_budget_gb: Optional[float] = None,
        context_tokens: int = 4096,
        target_format: str = "auto",
        solver_kind: str = "greedy",
        hit_rate: float = 0.9,
    ) -> PlannerResult:
        self._last_model = model
        decisions: List[PlanDecision] = []

        # Step 1 — Capability discovery
        device = self.db.get(device_name)
        budget_gb = memory_budget_gb or device.total_addressable_gb or 8.0
        decisions.append(PlanDecision(1, "Capability discovery",
                                      f"{device.label} ({device.device_type})",
                                      f"RAM={device.ram_gb}GB VRAM={device.vram_gb}GB bw={device.memory_bw_gbps}GB/s"))

        # Step 2 — Cost model
        decisions.append(PlanDecision(2, "Cost model", "RAM + IO + compute + dequant",
                                      f"storage BW {device.storage_read_mbps}MB/s; compute {device.best_tflops:.1f} TFLOPS"))

        # Step 3 — Candidate generation
        candidates = self.solver.build_candidates(model)
        decisions.append(PlanDecision(3, "Candidate generation",
                                      f"{len(model.tensors)} tensors × ≤9 formats",
                                      "Pareto-filtered non-dominated candidates"))

        # Step 4 — Static Oracle (a priori filtering)
        # norms always FP32, embeddings keep ≥Q8 when budget permits
        imp = {"norm": 99.0, "embed": 1.4, "lm_head": 1.5, "moe_gate": 1.2}
        candidates = self.solver.build_candidates(model, importance_map=imp)
        decisions.append(PlanDecision(4, "Static Oracle", "pin norms=FP32; boost embed/lm_head/gate importance",
                                      "known sensitivity hierarchy from earlier validations"))

        # Step 5 — Prediction of naive options
        budget_bytes = int(budget_gb * (1024 ** 3))
        naives = {fmt: self.solver.uniform_plan(model, fmt) for fmt in ("Q4_0", "Q8_0", "Q2_K") if self.db.names()}
        pred_lines = "; ".join(f"{n}:{p.total_size_gb:.1f}GB,{p.average_bpw:.2f}bpw" for n, p in naives.items())
        decisions.append(PlanDecision(5, "Prediction", pred_lines,
                                      "uniform baselines for later mixed vs uniform comparison"))

        # Step 6 — Allocation
        kv_prec_hint = "int4" if budget_gb < 10 else "fp16"
        kv_gb_res, _ = MemoryProfiler.kv_cache_gb(model, context_tokens, kv_prec_hint)
        act_gb_res = MemoryProfiler.activations_gb(model, context_tokens)
        oh_gb = max(0.5, 0.08 * (device.total_addressable_gb or 8.0))
        exp_cache = 0.5 if model.is_moe else 0.0
        reserved_gb = kv_gb_res + act_gb_res + oh_gb + exp_cache + 0.15
        storage_budget_gb = max(budget_gb - reserved_gb, budget_gb * 0.5)
        budget_bytes = int(storage_budget_gb * (1024 ** 3))
        if target_format == "auto":
            if solver_kind == "dp":
                plan = self.solver.solve_dp(model, budget_bytes, candidates)
            else:
                plan = self.solver.solve_greedy(model, budget_bytes, candidates)
        else:
            plan = self.solver.uniform_plan(model, target_format)
        decisions.append(PlanDecision(6, "Allocation", f"{plan.name}: {plan.average_bpw:.2f} avg bpw",
                                      f"{plan.total_size_gb:.2f} GB of {budget_gb:.1f} GB memory reserved {reserved_gb:.2f} GB"))

        # Step 7 — Execution planning
        streaming_needed = plan.total_size_gb > budget_gb or model.is_moe
        resident = [t.name for t in (model.dense_tensors() if streaming_needed and model.is_moe else model.tensors)]
        streaming = [t.name for t in (model.expert_tensors() if streaming_needed and model.is_moe else [])]
        kv_prec = "int4" if budget_gb < 10 else "fp16"
        cache_cap = min(max(model.num_experts // 4, 1), 32)
        placements = {
            "norm": "cpu",
            "embed": "gpu" if device.has_cuda else ("htp" if device.has_htp else "cpu"),
            "attn_qkv": "gpu" if device.has_cuda else "cpu",
            "attn_out": "gpu" if device.has_cuda else "cpu",
            "mlp": "gpu" if device.has_cuda else ("htp" if device.has_htp else "cpu"),
            "moe_expert": "gpu" if device.has_cuda else ("htp" if device.has_htp else "cpu"),
            "lm_head": "gpu" if device.has_cuda else "cpu",
        }
        decisions.append(PlanDecision(7, "Execution planning",
                                      f"{len(resident)} resident / {len(streaming)} streamed tensors",
                                      f"KV={kv_prec}, expert cache cap={cache_cap}, placements={dict(list(placements.items())[:3])}"))

        # Memory profile & cost
        mem = self.profiler.profile(model, device, quant_plan=plan.plan, resident_tensors=resident,
                                    context_tokens=context_tokens, kv_precision=kv_prec,
                                    memory_budget_gb=budget_gb)
        cost = self.cost.t_token(model, device, plan=plan, hit_rate=hit_rate)

        strategy = ExecutionStrategy(
            resident_tensors=resident,
            streaming_tensors=streaming,
            kv_precision=kv_prec,
            expert_cache_capacity_experts=cache_cap,
            backend_placements=placements,
            streaming_enabled=streaming_needed,
            estimated_decode_tok_s=cost.decode_tok_s,
            estimated_working_set_gb=mem.working_set_gb,
        )

        res = PlannerResult(
            model_name=model.name,
            device_name=device.name,
            allocation_plan=plan,
            memory_profile=mem,
            cost_breakdown=cost,
            execution_strategy=strategy,
            decisions=decisions,
        )
        self.last_result = res
        return res

    # Step 8 — Runtime feedback
    def feedback(self, measured: dict) -> PlannerResult:
        """measured: {"measured_decode_tok_s": ..., "measured_working_set_gb": ..., "fallback_pct": ...}."""
        if self.last_result is None:
            raise RuntimeError("Run planner first")
        r = self.last_result
        r.measured_feedback = measured
        pred = r.cost_breakdown.decode_tok_s
        meas = measured.get("measured_decode_tok_s", pred)
        gap = (meas - pred) / max(pred, 1e-9) * 100
        r.decisions.append(PlanDecision(8, "Runtime feedback",
                                        f"measured {meas} tok/s vs pred {pred} ({gap:+.0f}%)",
                                        "feed into adaptation if gap > 25%"))
        if abs(gap) > 25 or measured.get("fallback_pct", 0) > 30:
            return self.adapt(measured)
        return r

    # Step 9 — Adaptive revision
    def adapt(self, measured: dict) -> PlannerResult:
        """Closed loop: retune device profile from measurements and re-run."""
        r = self.last_result
        device = self.db.get(r.device_name)
        if "fallback_pct" in measured:
            device.cpu_fallback_risk = measured["fallback_pct"] / 100.0
        # storage BW calibration if measured
        if "measured_storage_mbps" in measured:
            device.storage_read_mbps = measured["measured_storage_mbps"]
        # re-run planner
        new_res = self.run(
            model=self._last_model,
            device_name=r.device_name,
        )
        new_res.decisions.append(PlanDecision(9, "Adaptive revision",
                                              f"recomputed plan with calibrated device costs",
                                              f"fallback risk now {device.cpu_fallback_risk:.0%}"))
        new_res.adapted = True
        return new_res

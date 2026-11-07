"""L1–L3 validation framework (Profiler Axe 11 + Section 8 success metrics).

Levels:
  L1 — Structure and reconstruction (weights, dims, formats, numerical errors)
  L2 — Operation behavior (activations, logits, critical ops)
  L3 — Model-level validation (perplexity, generation quality, performance)

Golden rule: "La réussite de L1 ne démontre pas la réussite de L3."
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional
from .profile_model import ModelMap
from .mixed_precision import AllocationPlan
from .profiler import MemoryProfile
from .cost_model import TokenCostBreakdown
from .capability_db import DeviceProfile


@dataclass
class CheckResult:
    level: str                        # "L1", "L2", "L3"
    name: str
    status: str                       # "PASS", "WARN", "FAIL"
    value: float
    threshold: float
    message: str


@dataclass
class ValidationReport:
    plan_name: str
    overall: str                      # "PASS", "WARN", "FAIL"
    checks: List[CheckResult]
    quality_summary: dict = field(default_factory=dict)
    memory_summary: dict = field(default_factory=dict)
    io_summary: dict = field(default_factory=dict)
    compute_summary: dict = field(default_factory=dict)
    perf_summary: dict = field(default_factory=dict)
    prediction_summary: dict = field(default_factory=dict)

    def passed(self) -> bool:
        return self.overall == "PASS"

    def to_dict(self) -> dict:
        return asdict(self)

    def export(self, path: str = "validation_report.json") -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)


class ValidationEngine:
    """Runs L1–L3 gates on a plan/model/device configuration or empirical measurements."""

    def __init__(self) -> None:
        pass

    def validate_plan(
        self,
        model: ModelMap,
        plan: AllocationPlan,
        mem: Optional[MemoryProfile] = None,
        cost: Optional[TokenCostBreakdown] = None,
        device: Optional[DeviceProfile] = None,
        max_weighted_error: float = 5e-2,
        max_ppl_delta_est: float = 0.5,
        min_tok_s: float = 5.0,
    ) -> ValidationReport:
        checks: List[CheckResult] = []

        # ----------------- L1 : Structure & reconstruction
        # (a) norms must stay FP32
        norms_ok = all(
            plan.per_tensor[t.name]["format"] == "FP32"
            for t in model.tensors if t.tensor_family == "norm"
        )
        checks.append(CheckResult(
            "L1", "norm_tensors_FP32",
            "PASS" if norms_ok else "FAIL", 1.0 if norms_ok else 0.0, 1.0,
            "All norm layers preserved at FP32" if norms_ok else "Some norm layers are quantized — instability risk",
        ))

        # (b) budget adherence
        budget_ratio = plan.total_size_bytes / max(plan.budget_bytes, 1)
        st = "PASS" if budget_ratio <= 1.0 else ("WARN" if budget_ratio <= 1.02 else "FAIL")
        checks.append(CheckResult("L1", "storage_within_budget", st, round(budget_ratio, 3), 1.0,
                                  f"checkpoint {plan.total_size_gb:.2f} GB vs budget {plan.budget_bytes/1024**3:.2f} GB"))

        # (c) reasonable average bpw
        checks.append(CheckResult("L1", "min_average_bpw", "PASS" if plan.average_bpw >= 1.0 else "FAIL",
                                  round(plan.average_bpw, 2), 1.0,
                                  f"average {plan.average_bpw:.2f} bpw"))

        # (d) weighted reconstruction error
        st = "PASS" if plan.total_weighted_error <= max_weighted_error else "WARN"
        checks.append(CheckResult("L1", "weighted_reconstruction_error", st,
                                  plan.total_weighted_error, max_weighted_error,
                                  f"weighted error {plan.total_weighted_error:.3e}"))

        # (e) all tensors present in plan
        present = len(plan.per_tensor) == len(model.tensors)
        checks.append(CheckResult("L1", "all_tensors_covered", "PASS" if present else "FAIL",
                                  1.0 if present else 0.0, 1.0, f"{len(plan.per_tensor)}/{len(model.tensors)} tensors"))

        # ----------------- L2 : Operation behavior (analytic proxies)
        cos_mean = sum(
            math.cos(math.atan(d["error"])) for d in plan.per_tensor.values()
        ) / max(len(plan.per_tensor), 1)
        checks.append(CheckResult("L2", "mean_cosine_similarity", "PASS" if cos_mean >= 0.985 else "WARN",
                                  round(cos_mean, 4), 0.985, f"mean cosine proxy {cos_mean:.4f}"))

        # logit agreement proxy — normalized by number of tensors
        mean_w_err = plan.total_weighted_error / max(len(plan.per_tensor), 1)
        logit_agr = max(0.0, 1.0 - mean_w_err * 25.0)
        checks.append(CheckResult("L2", "logit_agreement_proxy", "PASS" if logit_agr >= 0.93 else "WARN",
                                  round(logit_agr, 3), 0.93, f"logit agreement ~{logit_agr:.3f}"))

        # token agreement proxy
        tok_agr = max(0.0, 1.0 - mean_w_err * 45.0)
        checks.append(CheckResult("L2", "token_agreement_proxy", "PASS" if tok_agr >= 0.88 else "WARN",
                                  round(tok_agr, 3), 0.88, f"token agreement ~{tok_agr:.3f}"))

        # ----------------- L3 : Model-level quality & performance
        # PPL delta estimate anchored to GTX 1080 validation (ΔPPL mixed vs Q4 ≈ 0.26)
        fp16_bpw = 16.0
        bpw_drop = max(fp16_bpw - plan.average_bpw, 0.0)
        ppl_delta = round(0.015 * bpw_drop + mean_w_err * 8.0, 3)
        checks.append(CheckResult("L3", "estimated_PPL_delta", "PASS" if ppl_delta <= max_ppl_delta_est else "WARN",
                                  ppl_delta, max_ppl_delta_est, f"estimated PPL delta vs FP16: +{ppl_delta}"))

        if cost is not None:
            st = "PASS" if cost.decode_tok_s >= min_tok_s else "WARN"
            checks.append(CheckResult("L3", "min_decode_throughput", st, cost.decode_tok_s, min_tok_s,
                                      f"{cost.decode_tok_s} tok/s decode"))

        if mem is not None:
            st = "PASS" if not mem.exceeds_budget else "FAIL"
            checks.append(CheckResult("L3", "working_set_within_memory", st, mem.working_set_gb,
                                      mem.memory_budget_gb,
                                      f"{mem.working_set_gb} GB / {mem.memory_budget_gb} GB budget"))

        if device is not None and device.cpu_fallback_risk > 0.3:
            checks.append(CheckResult("L3", "accelerator_fallback_risk", "WARN",
                                      device.cpu_fallback_risk * 100, 30.0,
                                      f"{device.cpu_fallback_risk*100:.0f}% CPU fallback risk on MUL_MAT"))

        fails = sum(1 for c in checks if c.status == "FAIL")
        warns = sum(1 for c in checks if c.status == "WARN")
        overall = "FAIL" if fails else ("WARN" if warns else "PASS")

        return ValidationReport(
            plan_name=plan.name,
            overall=overall,
            checks=checks,
            quality_summary={
                "estimated_ppl_delta": ppl_delta,
                "mean_cosine": round(cos_mean, 4),
                "logit_agreement": round(logit_agr, 3),
                "weighted_reconstruction_error": plan.total_weighted_error,
            },
            memory_summary={
                "storage_gb": round(plan.total_size_gb, 2),
                "working_set_gb": mem.working_set_gb if mem else None,
                "kv_gb": mem.kv_cache_gb if mem else None,
                "exceeds_budget": mem.exceeds_budget if mem else None,
            },
            io_summary={
                "cold_bytes_tok_mb": round(mem.cold_bytes_per_token / (1024 ** 2), 2) if mem else None,
                "io_dominated": cost.io_dominated if cost else None,
            },
            compute_summary={
                "t_compute_ms": cost.t_compute_ms if cost else None,
                "t_dequant_ms": cost.t_dequant_ms if cost else None,
                "cpu_fallback_risk": device.cpu_fallback_risk if device else None,
            },
            perf_summary={
                "decode_tok_s": cost.decode_tok_s if cost else None,
                "prefill_tok_s": cost.prefill_tok_s if cost else None,
            },
            prediction_summary={
                "confidence_interval": "calibrated 10%" if (device and device.validated) else "uncalibrated 30-40%",
            },
        )

    @staticmethod
    def print(rep: ValidationReport) -> str:
        icons = {"PASS": "OK", "WARN": "WARN", "FAIL": "FAIL"}
        lines = [
            f"=== Validation Report : {rep.plan_name} — OVERALL: {rep.overall} ===",
            "Level  Check                        Status  Value      Thresh    Note",
            "-" * 80,
        ]
        for c in rep.checks:
            lines.append(f"{c.level:5}  {c.name:<28} {c.status:6}  {c.value:9.3f} {c.threshold:8.3f}  {c.message}")
        lines.append("")
        lines.append(f"  Quality     : est. ΔPPL {rep.quality_summary.get('estimated_ppl_delta')} | "
                     f"cos {rep.quality_summary.get('mean_cosine')} | logit agree {rep.quality_summary.get('logit_agreement')}")
        lines.append(f"  Memory      : storage {rep.memory_summary.get('storage_gb')} GB | "
                     f"working set {rep.memory_summary.get('working_set_gb')} GB")
        lines.append(f"  I/O         : cold IO {rep.io_summary.get('cold_bytes_tok_mb')} MB/tok | "
                     f"bound={'I/O' if rep.io_summary.get('io_dominated') else 'compute'}")
        lines.append(f"  Performance : decode {rep.perf_summary.get('decode_tok_s')} tok/s | "
                     f"prefill {rep.perf_summary.get('prefill_tok_s')} tok/s")
        return "\n".join(lines)

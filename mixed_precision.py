"""Mixed-precision allocation (Profiler Axe 4 + Quant Algo 1, 4, 10).

Selects per-tensor precision to minimize Σ w_i * E_{i,q} subject to Σ S_{i,q} ≤ B,
implementing convex-hull greedy, dynamic-programming, and Pareto sweep solvers,
and exporting quantization_plan.json.
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from .profile_model import ModelMap, TensorMeta
from .quant_formats import QuantFormat, FormatRegistry, quant_registry
from .sensitivity_engine import SensitivityEngine
from .quantization_engine import QuantizationEngine, QuantCandidate


@dataclass
class AllocationPlan:
    name: str
    model_name: str
    solver: str
    budget_bytes: int
    total_size_bytes: int
    total_weighted_error: float
    average_bpw: float
    per_tensor: Dict[str, dict]        # name -> {format, size, bpw, error, justification}
    plan: Dict[str, str]              # name -> format (for quick lookup)

    @property
    def total_size_gb(self) -> float:
        return self.total_size_bytes / (1024 ** 3)

    @property
    def headroom_mb(self) -> float:
        return (self.budget_bytes - self.total_size_bytes) / (1024 ** 2)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "model_name": self.model_name,
            "solver": self.solver,
            "budget_bytes": self.budget_bytes,
            "budget_gb": round(self.budget_bytes / (1024 ** 3), 3),
            "total_size_bytes": self.total_size_bytes,
            "total_size_gb": round(self.total_size_gb, 3),
            "average_bpw": round(self.average_bpw, 3),
            "total_weighted_error": self.total_weighted_error,
            "per_tensor": self.per_tensor,
        }

    def export(self, path: str = "quantization_plan.json") -> None:
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=1)

    def summary(self) -> str:
        formats_count: Dict[str, int] = {}
        for d in self.per_tensor.values():
            fmt = d["format"]
            formats_count[fmt] = formats_count.get(fmt, 0) + 1
        return (
            f"[{self.name}] size={self.total_size_gb:.2f} GB | avg {self.average_bpw:.2f} bpw | "
            f"weighted error={self.total_weighted_error:.4e} | headroom={self.headroom_mb:.0f} MB | "
            + ", ".join(f"{k}:{v}" for k, v in sorted(formats_count.items()))
        )


class MixedPrecisionSolver:
    """Solves the multi-choice knapsack allocation problem."""

    def __init__(self, registry: Optional[FormatRegistry] = None) -> None:
        self.registry = registry or quant_registry
        self.engine = QuantizationEngine(self.registry)

    # ------------------------------------------------ candidate generation
    def build_candidates(
        self,
        model: ModelMap,
        formats: Optional[List[str]] = None,
        importance_map: Optional[Dict[str, float]] = None,
    ) -> Dict[str, List[QuantCandidate]]:
        """Return per-tensor non-dominated candidates."""
        imp = importance_map or {}
        out: Dict[str, List[QuantCandidate]] = {}
        for t in model.tensors:
            if t.tensor_family == "norm":
                fmt = self.registry.get("FP32")
                sm = SensitivityEngine.estimate(t, fmt)
                out[t.name] = [QuantCandidate("FP32", fmt.quantized_bytes(t.n_params),
                                              32.0, sm.normalized_l2, sm.activation_weighted_error,
                                              fmt.dequant_ns_per_weight)]
                continue
            cands = self.engine.candidates(t, formats=formats, importance=imp.get(t.tensor_family, 1.0))
            out[t.name] = [c for c in cands if not c.dominated]
        return out

    # ------------------------------------------------ greedy solver
    def solve_greedy(
        self,
        model: ModelMap,
        budget_bytes: int,
        candidates: Optional[Dict[str, List[QuantCandidate]]] = None,
        name: str = "mixed_greedy",
    ) -> AllocationPlan:
        """Convex-hull marginal-gain greedy solver: O(T*F*log(T*F))."""
        cands = candidates or self.build_candidates(model)

        # Per tensor: lower convex hull (size vs error) -> diminishing marginal gains
        hulls: Dict[str, List[QuantCandidate]] = {}
        for nm, cs in cands.items():
            sorted_cs = sorted(cs, key=lambda c: (c.size_bytes, c.weighted_error))
            hull: List[QuantCandidate] = []
            for c in sorted_cs:
                while len(hull) >= 2:
                    p1, p2 = hull[-2], hull[-1]
                    dx1 = p2.size_bytes - p1.size_bytes
                    dy1 = p1.weighted_error - p2.weighted_error
                    dx2 = c.size_bytes - p2.size_bytes
                    dy2 = p2.weighted_error - c.weighted_error
                    if dx1 <= 0 or (dx2 > 0 and dy2 * dx1 >= dy1 * dx2):
                        hull.pop()
                    else:
                        break
                hull.append(c)
            hulls[nm] = hull

        # Marginal steps: (gain, tensor, new_idx, delta_size)
        steps: List[Tuple[float, str, int, int]] = []
        for nm, hull in hulls.items():
            for i in range(len(hull) - 1):
                ds = hull[i + 1].size_bytes - hull[i].size_bytes
                de = hull[i].weighted_error - hull[i + 1].weighted_error
                if ds > 0 and de > 0:
                    steps.append((de / ds, nm, i + 1, ds))
        steps.sort(key=lambda s: s[0], reverse=True)

        cur_idx: Dict[str, int] = {nm: 0 for nm in hulls}
        cur_size = sum(hulls[nm][0].size_bytes for nm in hulls)
        for gain, nm, new_idx, ds in steps:
            if new_idx != cur_idx[nm] + 1:
                continue
            if cur_size + ds > budget_bytes:
                continue
            cur_size += ds
            cur_idx[nm] = new_idx

        orig_idx: Dict[str, int] = {}
        for nm, hi in cur_idx.items():
            chosen = hulls[nm][hi]
            for j, c in enumerate(cands[nm]):
                if c.format_name == chosen.format_name:
                    orig_idx[nm] = j
                    break
        return self._make_plan(name, model, solver="greedy", budget_bytes=budget_bytes,
                                cands=cands, idx_map=orig_idx)

    # ------------------------------------------------ dynamic programming (multiple-choice knapsack)
    def solve_dp(
        self,
        model: ModelMap,
        budget_bytes: int,
        candidates: Optional[Dict[str, List[QuantCandidate]]] = None,
        bins: int = 500,
        name: str = "mixed_dp",
    ) -> AllocationPlan:
        """Exact DP on discretized budget (scale to ~bins integer units)."""
        cands = candidates or self.build_candidates(model)
        names = list(cands.keys())
        unit = max(1, budget_bytes // bins)

        dp: Dict[int, Tuple[float, Dict[str, int]]] = {0: (0.0, {})}
        for nm in names:
            cs = cands[nm]
            next_dp: Dict[int, Tuple[float, Dict[str, int]]] = {}
            for w, (err, path) in dp.items():
                for i, c in enumerate(cs):
                    nw = w + math.ceil(c.size_bytes / unit)
                    if nw > bins:
                        continue
                    nerr = err + c.weighted_error
                    if nw not in next_dp or nerr < next_dp[nw][0]:
                        npath = dict(path)
                        npath[nm] = i
                        next_dp[nw] = (nerr, npath)
            dp = next_dp

        best = min(dp.values(), key=lambda x: x[0])
        return self._make_plan(name, model, solver="dp", budget_bytes=budget_bytes,
                                cands=cands, idx_map=best[1])

    # ------------------------------------------------ Pareto frontier sweep
    def solve_pareto(
        self,
        model: ModelMap,
        n_points: int = 10,
        min_budget_gb: Optional[float] = None,
        max_budget_gb: Optional[float] = None,
    ) -> List[AllocationPlan]:
        """Sweep budget to produce the (size, quality) Pareto frontier."""
        cands = self.build_candidates(model)
        min_b = sum(min(c.size_bytes for c in cs) for cs in cands.values())
        max_b = sum(max(c.size_bytes for c in cs) for cs in cands.values())
        if min_budget_gb is not None:
            min_b = max(min_b, int(min_budget_gb * (1024 ** 3)))
        if max_budget_gb is not None:
            max_b = min(max_b, int(max_budget_gb * (1024 ** 3)))

        plans: List[AllocationPlan] = []
        for i in range(n_points):
            b = min_b + (max_b - min_b) * i // max(n_points - 1, 1)
            p = self.solve_greedy(model, b, cands, name=f"pareto_{i}")
            plans.append(p)
        return plans

    # ------------------------------------------------ uniform baseline plans (for comparison)
    def uniform_plan(
        self,
        model: ModelMap,
        format_name: str,
        name: Optional[str] = None,
        importance_map: Optional[Dict[str, float]] = None,
    ) -> AllocationPlan:
        """Uniform single-format plan (with norms kept FP32)."""
        imp = importance_map or {}
        fmt = self.registry.get(format_name)
        per_tensor: Dict[str, dict] = {}
        total_size = 0
        total_err = 0.0
        total_bits = 0
        total_params = 0
        plan: Dict[str, str] = {}
        for t in model.tensors:
            f = self.registry.get("FP32") if t.tensor_family == "norm" else fmt
            sm = SensitivityEngine.estimate(t, f, importance_override=imp.get(t.tensor_family, 1.0))
            sz = f.quantized_bytes(t.n_params)
            per_tensor[t.name] = {
                "format": f.name,
                "size_bytes": sz,
                "bpw": f.effective_bpw,
                "error": sm.normalized_l2,
                "weighted_error": sm.activation_weighted_error,
                "justification": f"Uniform {format_name} baseline",
            }
            plan[t.name] = f.name
            total_size += sz
            total_err += sm.activation_weighted_error
            total_bits += t.n_params * f.effective_bpw
            total_params += t.n_params
        return AllocationPlan(
            name=name or f"uniform_{format_name}",
            model_name=model.name,
            solver="uniform",
            budget_bytes=total_size,
            total_size_bytes=total_size,
            total_weighted_error=total_err,
            average_bpw=total_bits / max(total_params, 1),
            per_tensor=per_tensor,
            plan=plan,
        )

    # ------------------------------------------------ helpers
    @staticmethod
    def _make_plan(
        name: str, model: ModelMap, solver: str, budget_bytes: int,
        cands: Dict[str, List[QuantCandidate]], idx_map: Dict[str, int],
    ) -> AllocationPlan:
        total_size = 0
        total_err = 0.0
        total_bits = 0
        total_params = 0
        per_tensor: Dict[str, dict] = {}
        plan: Dict[str, str] = {}
        tensor_by_name = {t.name: t for t in model.tensors}
        for nm, i in idx_map.items():
            c = cands[nm][i]
            t = tensor_by_name[nm]
            just = f"Sensitivity {c.error:.4f}, importance {c.weighted_error/max(c.error,1e-9):.2f}"
            per_tensor[nm] = {
                "format": c.format_name,
                "size_bytes": c.size_bytes,
                "bpw": c.bpw,
                "error": c.error,
                "weighted_error": c.weighted_error,
                "justification": just,
            }
            plan[nm] = c.format_name
            total_size += c.size_bytes
            total_err += c.weighted_error
            total_bits += t.n_params * c.bpw
            total_params += t.n_params
        return AllocationPlan(
            name=name,
            model_name=model.name,
            solver=solver,
            budget_bytes=budget_bytes,
            total_size_bytes=total_size,
            total_weighted_error=total_err,
            average_bpw=total_bits / max(total_params, 1),
            per_tensor=per_tensor,
            plan=plan,
        )

    @staticmethod
    def compare(plans: List[AllocationPlan]) -> str:
        """Side-by-side comparison table."""
        header = f"{'Plan':<18} {'Size (GB)':>10} {'Avg bpw':>8} {'W. Error':>12} {'Formats used'}"
        lines = [header, "-" * len(header)]
        for p in plans:
            fc: Dict[str, int] = {}
            for d in p.per_tensor.values():
                fc[d["format"]] = fc.get(d["format"], 0) + 1
            fstr = ", ".join(f"{k}:{v}" for k, v in sorted(fc.items()))
            lines.append(f"{p.name:<18} {p.total_size_gb:>10.2f} {p.average_bpw:>8.2f} "
                         f"{p.total_weighted_error:>12.4e} {fstr}")
        return "\n".join(lines)

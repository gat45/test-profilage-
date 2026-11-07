"""Hardware cost model (Profiler Axe 10, Axe 12 — Latency & performance prediction).

Computes T_token = T_io + T_compute + T_dequant + T_sync + T_other with overlap,
converting to tok/s for decode and prefill phases, cold vs hot cache.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from .profile_model import ModelMap
from .capability_db import DeviceProfile
from .quant_formats import QuantFormat, FormatRegistry, quant_registry
from .mixed_precision import AllocationPlan


@dataclass
class TokenCostBreakdown:
    t_io_ms: float
    t_compute_ms: float
    t_dequant_ms: float
    t_sync_ms: float
    t_other_ms: float
    t_effective_ms: float
    decode_tok_s: float
    prefill_tok_s: float
    cold_vs_hot_ratio: float          # hot tok/s / cold tok/s
    io_dominated: bool
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


class CostModel:
    """Predicts per-token time breakdown and throughput."""

    def __init__(self, registry: Optional[FormatRegistry] = None) -> None:
        self.registry = registry or quant_registry

    def t_token(
        self,
        model: ModelMap,
        device: DeviceProfile,
        plan: Optional[AllocationPlan] = None,
        hit_rate: float = 1.0,                    # MoE expert cache hit rate
        prompt_len: int = 512,
        overlap_alpha: float = 0.6,               # IO/compute overlap [0, 1]
        batch_size: int = 1,
    ) -> TokenCostBreakdown:
        """Decode step time breakdown for one generated token."""

        # (1) T_io : bytes to read from storage
        if model.is_moe:
            moe_bytes = sum(t.size_bytes for t in model.expert_tensors())
            per_expert = moe_bytes / max(model.num_experts, 1)
            miss_bytes = model.top_k * per_expert * (1.0 - hit_rate)
        else:
            miss_bytes = 0.0                      # dense model assumed resident
        io_bw_mbps = max(device.storage_read_mbps, 50.0)
        t_io_s = (miss_bytes / (1024 ** 2)) / io_bw_mbps
        t_io_ms = t_io_s * 1000.0

        # (2) T_compute : 2 FLOPs per active parameter per token
        active_params = model.n_params_per_token_decode * batch_size
        gemm_flops = 2.0 * active_params
        # Auto-regressive decode (batch=1 GEMV) achieves ~45-60% of peak compute on memory-bound workloads
        gemv_efficiency = 0.55
        tflops = max(device.best_tflops * gemv_efficiency, 0.05)
        # Apply CPU fallback penalty if accelerator fails
        fb_factor = 1.0 + device.cpu_fallback_risk * 15.0  # fallback to CPU is ~15-30x slower
        t_comp_s = (gemm_flops / (tflops * 1e12)) * fb_factor
        t_comp_ms = t_comp_s * 1000.0

        # (3) T_dequant : per-format dequantization cost for active tensors
        t_deq_ms = 0.0
        plan_fmt = plan.plan if plan else {}
        # GPU with fused quant kernels is ~100x faster at dequant than CPU; NPU native formats are free
        scale = 0.01 if device.has_cuda else (0.05 if device.has_opencl else 1.0)
        for t in model.tensors:
            if t.is_moe:
                continue                          # MoE handled through dequant at stream-in
            fmt_name = plan_fmt.get(t.name, t.dtype)
            try:
                fmt = self.registry.get(fmt_name)
                cost_per = fmt.dequant_ns_per_weight * scale
                if device.has_htp and fmt.npu_ok:
                    cost_per = 0.0                # native HTP formats need no dequant step
                t_deq_ms += t.n_params * cost_per / 1e6
            except KeyError:
                pass

        # (4) T_sync : transfers host <-> accelerator + RPC
        transfer_bw = max(device.transfer_bw_gbps, 1.0)
        xfer_bytes = miss_bytes if device.has_cuda else 0.0
        t_sync_ms = (xfer_bytes / (transfer_bw * 1e9)) * 1000.0 + device.rpc_overhead_ms

        # (5) T_other : per-layer attention + RMSNorm + kernel launch (~0.25 ms/layer) + sampling/framework
        layer_overhead = 0.22 * model.num_hidden_layers
        t_other_ms = 1.2 + layer_overhead

        # Overlapped effective time
        t_eff_ms = max(t_io_ms, t_comp_ms) + (1.0 - overlap_alpha) * min(t_io_ms, t_comp_ms)
        t_eff_ms += t_deq_ms + t_sync_ms + t_other_ms

        # Decode throughput
        decode = 1000.0 / max(t_eff_ms, 0.05)

        # Prefill : full model loaded once, GEMM over prompt
        prefill_flops = 2.0 * model.total_params * prompt_len
        t_pre_comp = (prefill_flops / (tflops * 1e12)) * 1000.0 * fb_factor
        t_pre_io = (plan.total_size_bytes if plan else model.total_params * 2) / (1024 ** 2) / io_bw_mbps * 1000.0
        prefill = prompt_len * 1000.0 / max(t_pre_comp + t_pre_io, 5.0)

        # Hot-cache decode (hit_rate = 1.0)
        hot = 1000.0 / max(t_comp_ms + t_deq_ms + t_other_ms, 0.05)
        ratio = hot / decode

        notes = ""
        if device.cpu_fallback_risk > 0.3:
            notes += f"Warning: {device.cpu_fallback_risk*100:.0f}% CPU fallback risk on MUL_MAT kernels. "
        if t_io_ms > t_comp_ms * 1.5:
            notes += "I/O bound: SSD streaming dominates — increase expert cache or compress. "
        if not device.validated:
            notes += "Device profile not yet empirically validated. "

        return TokenCostBreakdown(
            t_io_ms=round(t_io_ms, 2),
            t_compute_ms=round(t_comp_ms, 2),
            t_dequant_ms=round(t_deq_ms, 2),
            t_sync_ms=round(t_sync_ms, 2),
            t_other_ms=round(t_other_ms, 2),
            t_effective_ms=round(t_eff_ms, 2),
            decode_tok_s=round(decode, 2),
            prefill_tok_s=round(prefill, 1),
            cold_vs_hot_ratio=round(ratio, 2),
            io_dominated=t_io_ms > t_comp_ms,
            notes=notes.strip(),
        )

    def print_breakdown(self, model_name: str, device_name: str, b: TokenCostBreakdown) -> str:
        pct = lambda x: f"{100*x/max(b.t_effective_ms, 1e-9):5.1f}%"
        return (
            f"=== Cost Model : {model_name} on {device_name} ===\n"
            f"  T_token (effective) : {b.t_effective_ms:.1f} ms  →  {b.decode_tok_s} tok/s decode | "
            f"{b.prefill_tok_s} tok/s prefill\n"
            f"    T_io      : {b.t_io_ms:>7.1f} ms ({pct(b.t_io_ms)})\n"
            f"    T_compute : {b.t_compute_ms:>7.1f} ms ({pct(b.t_compute_ms)})\n"
            f"    T_dequant : {b.t_dequant_ms:>7.1f} ms ({pct(b.t_dequant_ms)})\n"
            f"    T_sync    : {b.t_sync_ms:>7.1f} ms ({pct(b.t_sync_ms)})\n"
            f"    T_other   : {b.t_other_ms:>7.1f} ms ({pct(b.t_other_ms)})\n"
            f"  hot/cold ratio: {b.cold_vs_hot_ratio:.1f}x  |  bound: {'I/O' if b.io_dominated else 'compute'}"
            + (f"\n  notes: {b.notes}" if b.notes else "")
        )

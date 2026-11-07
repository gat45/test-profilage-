"""Qualcomm HTP trace analyzer (Profiler Axe 7 — CPU/GPU/NPU profiling).

Parses QAIRT / Hexagon execution traces, extracts VTCM usage, DMA transfers,
per-layer placement (HTP vs CPU), and computes the MUL_MAT CPU-fallback
percentage — the exact mechanism that caused Bonsai-27B to drop from 13.9 tok/s
estimated to 0.32 tok/s measured.
"""

from __future__ import annotations
import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from .capability_db import DeviceProfile


@dataclass
class HexagonTrace:
    raw_source: str
    n_nodes_total: int
    n_nodes_htp: int
    n_nodes_cpu: int
    mulmat_total: int
    mulmat_cpu: int
    cpu_fallback_pct: float           # 0-100
    vtcm_peak_mb: float
    dma_mb: float
    rpc_ms: float
    total_time_ms: float
    layer_placement: Dict[str, str]   # layer_name -> "HTP"|"CPU"
    notes: List[str]

    def to_dict(self) -> dict:
        return asdict(self)


class HexagonTraceParser:
    """Parses real or synthetic QAIRT Hexagon execution logs."""

    # Patterns matching typical QAIRT verbose logs
    NODE_PAT = re.compile(
        r"\[QAIRT\]\s+node=(\S+)\s+placed=(\S+)\s+time=([\d.]+)\s*ms(?:\s+vtcm=([\d.]+)\s*MB)?",
        re.IGNORECASE,
    )
    DMA_PAT = re.compile(r"DMA\s+(\S+)\s+bytes=([\d.]+)\s*MB", re.IGNORECASE)
    RPC_PAT = re.compile(r"RPC\s+([\d.]+)\s*ms", re.IGNORECASE)

    def parse(self, log_text: str, source: str = "log") -> HexagonTrace:
        nodes_htp = nodes_cpu = 0
        mulmat_total = mulmat_cpu = 0
        vtcm_peak = 0.0
        dma_mb = 0.0
        rpc_ms = 0.0
        total_time = 0.0
        placement: Dict[str, str] = {}
        notes: List[str] = []

        for line in log_text.splitlines():
            m = self.NODE_PAT.search(line)
            if m:
                node, placed, tms, v = m.group(1), m.group(2).upper(), float(m.group(3)), m.group(4)
                total_time += tms
                is_mulmat = "mul_mat" in node.lower() or "gemm" in node.lower()
                if "HTP" in placed:
                    nodes_htp += 1
                    placement[node] = "HTP"
                else:
                    nodes_cpu += 1
                    placement[node] = "CPU"
                    if is_mulmat:
                        mulmat_cpu += 1
                if is_mulmat:
                    mulmat_total += 1
                if v:
                    vtcm_peak = max(vtcm_peak, float(v))
                continue
            m = self.DMA_PAT.search(line)
            if m:
                dma_mb += float(m.group(2))
                continue
            m = self.RPC_PAT.search(line)
            if m:
                rpc_ms += float(m.group(1))

        total_nodes = nodes_htp + nodes_cpu
        fb_pct = 100.0 * mulmat_cpu / max(mulmat_total, 1)

        if fb_pct > 50:
            notes.append(f"CRITICAL: {fb_pct:.1f}% MUL_MAT kernels fell back to CPU — measured tok/s will be much lower than estimated.")
        if vtcm_peak > 7.9:
            notes.append(f"VTCM pressure: peak {vtcm_peak:.1f} MB near 8 MB limit.")

        return HexagonTrace(
            raw_source=source,
            n_nodes_total=total_nodes,
            n_nodes_htp=nodes_htp,
            n_nodes_cpu=nodes_cpu,
            mulmat_total=mulmat_total,
            mulmat_cpu=mulmat_cpu,
            cpu_fallback_pct=round(fb_pct, 1),
            vtcm_peak_mb=round(vtcm_peak, 2),
            dma_mb=round(dma_mb, 1),
            rpc_ms=round(rpc_ms, 2),
            total_time_ms=round(total_time, 2),
            layer_placement=placement,
            notes=notes,
        )

    # ------------------------------------------------ synthetic traces (document-anchored)
    @classmethod
    def bonsai_stress_trace(cls) -> str:
        """Synthetic trace reproducing the documented 99.7% fallback on Bonsai-27B."""
        lines = ["[QAIRT] Hexagon execution trace — Bonsai-27B on HTP v81"]
        for i in range(100):
            node = f"model.layers.{i//2}.mul_mat_{i%2}"
            if i == 0:
                lines.append(f"[QAIRT] node={node} placed=HTP v81 time=0.8ms vtcm=7.8MB")
            else:
                lines.append(f"[QAIRT] node={node} placed=CPU time=12.1ms")
        lines.append("DMA host->dsp bytes=400.0MB")
        lines.append("RPC 3.2ms")
        return "\n".join(lines)

    @classmethod
    def healthy_trace(cls, n_layers: int = 32) -> str:
        """Synthetic healthy trace (all layers on HTP)."""
        lines = ["[QAIRT] Hexagon execution trace — healthy"]
        for i in range(n_layers):
            node = f"model.layers.{i}.gemm"
            lines.append(f"[QAIRT] node={node} placed=HTP v81 time=0.6ms vtcm=5.2MB")
        lines.append("DMA host->dsp bytes=80.0MB")
        lines.append("RPC 0.9ms")
        return "\n".join(lines)

    @staticmethod
    def calibrate_device(trace: HexagonTrace, device: DeviceProfile) -> DeviceProfile:
        """Feed the empirical fallback percentage back into the device profile (closed loop)."""
        device.cpu_fallback_risk = trace.cpu_fallback_pct / 100.0
        return device

    @staticmethod
    def print(t: HexagonTrace) -> str:
        return (
            f"=== Hexagon HTP Trace Analysis ({t.raw_source}) ===\n"
            f"  nodes: {t.n_nodes_htp}/{t.n_nodes_total} on HTP | "
            f"MUL_MAT fallback to CPU: {t.mulmat_cpu}/{t.mulmat_total} ({t.cpu_fallback_pct:.1f}%)\n"
            f"  VTCM peak: {t.vtcm_peak_mb:.1f} MB (cap 8 MB) | DMA: {t.dma_mb:.0f} MB | RPC: {t.rpc_ms:.1f} ms | "
            f"total exec: {t.total_time_ms:.0f} ms\n"
            f"  placement: HTP={sum(1 for v in t.layer_placement.values() if v=='HTP')} / "
            f"CPU={sum(1 for v in t.layer_placement.values() if v=='CPU')}\n"
            + "".join(f"  note: {n}\n" for n in t.notes)
        )

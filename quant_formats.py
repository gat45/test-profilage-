"""Numerical format registry (Profiler Axe 2 — Numerical format analysis).

Defines the physical representation and cost model for each format the project
handles (effective bpw, block size, scale overhead, dequantization cost,
backend compatibility).
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class QuantFormat:
    name: str
    label: str
    effective_bpw: float              # bits per weight, overhead included
    block_size: int                   # number of values per block
    scale_bytes: int                  # scale metadata bytes per block
    category: str                     # "fp", "int", "gguf", "research"
    dequant_ns_per_weight: float = 0.5  # cost on CPU (dequant + copy)
    cpu_ok: bool = True
    gpu_ok: bool = True
    npu_ok: bool = False
    notes: str = ""

    def quantized_bytes(self, n_params: int) -> int:
        """Physical bytes needed to store n_params in this format."""
        if self.block_size <= 1:
            return math.ceil(n_params * self.effective_bpw / 8.0)
        n_blocks = math.ceil(n_params / self.block_size)
        raw_bits = self.block_size * self.effective_bpw
        return math.ceil(n_blocks * raw_bits / 8.0)

    def bpw(self) -> float:
        return self.effective_bpw

    def compatible_with(self, device_type: str) -> bool:
        if device_type == "cpu":
            return self.cpu_ok
        if device_type == "gpu":
            return self.gpu_ok
        if device_type in ("npu", "soc"):
            return self.npu_ok
        return False

    def to_dict(self) -> dict:
        return asdict(self)


class FormatRegistry:
    """Registry of numerical formats with property queries."""

    def __init__(self) -> None:
        self._formats: Dict[str, QuantFormat] = {}
        for f in BUILTIN_FORMATS:
            self.register(f)

    def register(self, fmt: QuantFormat) -> None:
        self._formats[fmt.name.lower()] = fmt

    def get(self, name: str) -> QuantFormat:
        key = name.lower()
        if key not in self._formats:
            raise KeyError(
                f"Unknown format '{name}'. Known: {list(self._formats.keys())}"
            )
        return self._formats[key]

    def __iter__(self):
        return iter(self._formats.values())

    def names(self) -> List[str]:
        return list(self._formats.keys())

    def export_report(self, path: str) -> None:
        """Writes format_capability_report.json."""
        with open(path, "w") as f:
            json.dump([f.to_dict() for f in self._formats.values()], f, indent=2)


# ---------------------------------------------------------------------------
# All formats from document Axe 2 + common variants
# ---------------------------------------------------------------------------
BUILTIN_FORMATS: List[QuantFormat] = [
    # Full precision
    QuantFormat("FP32", "IEEE FP32", 32.0, 1, 0, "fp", 0.05,
                notes="Reference — never used in deployment checkpoints."),
    QuantFormat("FP16", "IEEE FP16", 16.0, 1, 0, "fp", 0.08,
                gpu_ok=True, npu_ok=True, notes="Ubiquitous GPU precision."),
    QuantFormat("BF16", "Bfloat16", 16.0, 1, 0, "fp", 0.08,
                gpu_ok=True, npu_ok=True, notes="Kimi K3 dense trunk reference format."),

    # Simple integer formats
    QuantFormat("INT8", "Symmetric INT8 (per-tensor)", 8.0, 1, 0, "int", 0.15,
                gpu_ok=True, npu_ok=True, notes="Hardware-friendly, low-dequant overhead."),
    QuantFormat("INT4", "Symmetric INT4 (per-tensor)", 4.0, 1, 0, "int", 0.25,
                gpu_ok=True, npu_ok=True, notes="Mobile NPU sweet spot."),
    QuantFormat("INT2", "Symmetric INT2 (per-tensor)", 2.0, 1, 0, "int", 0.35,
                gpu_ok=True, npu_ok=False, notes="Aggressive research format."),

    # GGUF k-quants (llama.cpp)
    QuantFormat("Q8_0", "GGUF Q8_0 (32-block INT8)", 8.5, 32, 2, "gguf", 0.20,
                npu_ok=True, notes="High-quality checkpoint format, ~8.5 bpw with scale overhead."),
    QuantFormat("Q6_K", "GGUF Q6_K", 6.5625, 256, 16, "gguf", 0.30,
                notes="Quality sweet spot for sensitive layers."),
    QuantFormat("Q5_K_M", "GGUF Q5_K_M", 5.5, 256, 16, "gguf", 0.35,
                notes="Good compromise format."),
    QuantFormat("Q4_0", "GGUF Q4_0 (32-block INT4)", 4.5, 32, 2, "gguf", 0.30,
                npu_ok=True, notes="Classic Q4 — 4.5 bpw effective with per-block scale."),
    QuantFormat("Q4_K_M", "GGUF Q4_K_M", 4.5, 256, 16, "gguf", 0.40,
                notes="K-quant variant of Q4."),
    QuantFormat("Q3_K_M", "GGUF Q3_K_M", 3.4375, 256, 14, "gguf", 0.50,
                notes="Aggressive intermediate format."),
    QuantFormat("Q2_K", "GGUF Q2_K", 2.5625, 256, 18, "gguf", 0.60,
                notes="For least-sensitive tensors in mixed allocation."),

    # Ultra-compact research format Q1_0 (Bonsai-27B validated)
    QuantFormat("Q1_0", "Q1_0 (128-element blocks, 18 bytes)", 1.125, 128, 2, "research",
                0.90, npu_ok=False,
                notes="Validated on Bonsai-27B (128 elements, 18 octets/block → 1.125 bpw). "
                      "Theoretical bpw differs from real cost (scale overhead, decode latency)."),

    # Microscaling MXFP4 (Kimi K3 expert target)
    QuantFormat("MXFP4", "Microscaling MXFP4 (OCP standard)", 4.0, 32, 1, "research",
                0.35, npu_ok=True,
                notes="Microscaling format used for Kimi K3 MoE experts. Distinguish MXFP4 "
                      "value modification from lossless physical compression on top."),

    # Exponent-only scale format
    QuantFormat("E8M0", "E8M0 scale-only format", 8.0, 32, 0, "research", 0.20,
                notes="Per-block scales storage in FP8 (exponent-only)."),

    # IQ importance-matrix formats
    QuantFormat("IQ4_XS", "GGUF IQ4_XS (imatrix)", 4.25, 256, 8, "gguf", 0.45,
                notes="Importance-matrix guided Q4."),
    QuantFormat("IQ2_XXS", "GGUF IQ2_XXS (imatrix)", 2.0625, 256, 2, "gguf", 0.70,
                notes="Extreme imatrix format for Pareto sweep experiments."),
]

quant_registry = FormatRegistry()

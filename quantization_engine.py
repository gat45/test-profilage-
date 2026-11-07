"""Quantization engine (Quant Algo 1, 2, 3, 6).

Implements real numeric quantize/dequantize kernels (INT8/4/2, Q1_0 sign+scale,
MXFP4 microscaling), candidate generation per tensor, and activation-aware
calibration (salient-weight protection akin to AWQ).
"""

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
from .profile_model import TensorMeta
from .quant_formats import QuantFormat, FormatRegistry, quant_registry
from .sensitivity_engine import SensitivityEngine, SensitivityMetrics


@dataclass
class QuantCandidate:
    format_name: str
    size_bytes: int
    bpw: float
    error: float                      # normalized L2 error estimate (E_i)
    weighted_error: float             # w_i * E_i
    dequant_ns_per_weight: float
    dominated: bool = False           # Pareto-dominated (dropped)


class QuantizationEngine:
    """Produces and evaluates candidate quantizations for each tensor."""

    def __init__(self, registry: Optional[FormatRegistry] = None) -> None:
        self.registry = registry or quant_registry

    # ------------------------------------------------ numeric kernels (numpy)
    @staticmethod
    def _block_symmetric(w: np.ndarray, bits: int, block_size: int = 32) -> np.ndarray:
        """Block-wise symmetric quantization with per-block scale."""
        flat = w.reshape(-1).astype(np.float64)
        pad = (-len(flat)) % block_size
        if pad:
            flat = np.concatenate([flat, np.zeros(pad)])
        blocks = flat.reshape(-1, block_size)
        qmax = 2 ** (bits - 1) - 1
        scales = np.max(np.abs(blocks), axis=1, keepdims=True) / qmax
        scales[scales == 0] = 1.0
        q = np.clip(np.round(blocks / scales), -qmax, qmax)
        dq = (q * scales).reshape(-1)
        if pad:
            dq = dq[:-pad]
        return dq.reshape(w.shape).astype(w.dtype)

    def quant_dequant_int(self, w: np.ndarray, bits: int, block_size: int = 32) -> np.ndarray:
        return self._block_symmetric(w, bits, block_size)

    def quant_dequant_q1_0(self, w: np.ndarray, block_size: int = 128) -> np.ndarray:
        """Q1_0 style: 1-bit sign per weight + 1 scale per 128-element block."""
        flat = w.reshape(-1).astype(np.float64)
        pad = (-len(flat)) % block_size
        if pad:
            flat = np.concatenate([flat, np.zeros(pad)])
        blocks = flat.reshape(-1, block_size)
        signs = np.where(blocks >= 0, 1.0, -1.0)
        # Optimal scale for sign quantization in L2 sense is mean absolute value
        scales = np.mean(np.abs(blocks), axis=1, keepdims=True)
        dq = (signs * scales).reshape(-1)
        if pad:
            dq = dq[:-pad]
        return dq.reshape(w.shape).astype(w.dtype)

    def quant_dequant_mxfp4(self, w: np.ndarray, block_size: int = 32) -> np.ndarray:
        """MXFP4 microscaling: per-32-element E8M0 scale, FP4-E2M1 values."""
        flat = w.reshape(-1).astype(np.float64)
        pad = (-len(flat)) % block_size
        if pad:
            flat = np.concatenate([flat, np.zeros(pad)])
        blocks = flat.reshape(-1, block_size)
        scales = np.max(np.abs(blocks), axis=1, keepdims=True) / 6.0
        scales[scales == 0] = 1.0
        scaled = blocks / scales
        # FP4 E2M1 codebook (15 values symmetric around 0)
        codebook = np.array([-6, -4, -3, -2, -1.5, -1, -0.5, 0,
                             0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
        idx = np.abs(scaled[:, :, None] - codebook).argmin(axis=-1)
        q = codebook[idx]
        dq = (q * scales).reshape(-1)
        if pad:
            dq = dq[:-pad]
        return dq.reshape(w.shape).astype(w.dtype)

    # ------------------------------------------------ candidate generation
    CANDIDATE_FORMATS = ["Q8_0", "Q6_K", "Q4_0", "Q3_K_M", "Q2_K", "Q1_0", "MXFP4", "INT8", "INT4"]

    def candidates(
        self,
        tensor: TensorMeta,
        formats: Optional[List[str]] = None,
        importance: float = 1.0,
    ) -> List[QuantCandidate]:
        """Generate size/error candidate points for a tensor across formats."""
        fmts = formats or self.CANDIDATE_FORMATS
        out: List[QuantCandidate] = []
        for name in fmts:
            try:
                fmt = self.registry.get(name)
            except KeyError:
                continue
            sm = SensitivityEngine.estimate(tensor, fmt, importance_override=importance)
            size = fmt.quantized_bytes(tensor.n_params)
            out.append(QuantCandidate(
                format_name=fmt.name,
                size_bytes=size,
                bpw=fmt.effective_bpw,
                error=sm.normalized_l2,
                weighted_error=sm.activation_weighted_error,
                dequant_ns_per_weight=fmt.dequant_ns_per_weight,
            ))
        return self._pareto_filter(out)

    @staticmethod
    def _pareto_filter(cands: List[QuantCandidate]) -> List[QuantCandidate]:
        """Mark dominated candidates (larger size AND worse quality than another)."""
        sorted_cands = sorted(cands, key=lambda c: (c.size_bytes, c.error))
        best_error = float("inf")
        for c in sorted_cands:
            if c.error <= best_error:
                best_error = c.error
            else:
                c.dominated = True
        return sorted_cands

    # ------------------------------------------------ AWQ-style activation calibration (Algo 2)
    def calibrate_awq(
        self,
        w: np.ndarray,
        x: np.ndarray,
        base_bits: int = 4,
        salient_fraction: float = 0.01,
        block_size: int = 32,
    ) -> Tuple[np.ndarray, dict]:
        """Quantize with salient-weight protection guided by activation magnitudes."""
        act_col = np.mean(np.abs(x.astype(np.float64)), axis=0)
        saliency = np.abs(w.astype(np.float64)) * act_col[None, :]
        k = max(1, int(salient_fraction * saliency.size))
        thresh = np.partition(saliency.reshape(-1), -k)[-k]
        salient = saliency >= thresh

        # Quantize base
        dq_base = self.quant_dequant_int(w, base_bits, block_size)
        # Salient weights kept at INT8
        dq_salient = self.quant_dequant_int(w, 8, block_size)
        mixed = np.where(salient, dq_salient, dq_base)

        diff = (w - mixed).astype(np.float64)
        rms = float(np.sqrt(np.mean(diff ** 2)))
        norm_orig = float(np.linalg.norm(w.astype(np.float64))) + 1e-12
        l2 = float(np.linalg.norm(diff)) / norm_orig
        return mixed, {
            "rms": rms,
            "norm_l2": l2,
            "salient_fraction": salient_fraction,
            "salient_weights": int(np.sum(salient)),
            "base_format": f"INT{base_bits}",
            "salient_format": "INT8",
        }

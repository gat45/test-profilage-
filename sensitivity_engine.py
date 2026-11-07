"""Quantization sensitivity engine (Profiler Axe 3 — Sensitivity analysis).

Computes RMS error, max error, cosine similarity, SNR, normalized L2 error
  E_i = ||W - Ŵ||_2 / (||W||_2 + eps)
and activation-weighted sensitivity, exporting layer_sensitivity.csv /
tensor_sensitivity.csv.
"""

from __future__ import annotations
import csv
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
import numpy as np
from .profile_model import ModelMap, TensorMeta
from .quant_formats import QuantFormat, quant_registry


@dataclass
class SensitivityMetrics:
    tensor_name: str
    format_name: str
    rms_error: float
    max_error: float
    cosine_sim: float
    snr_db: float
    normalized_l2: float          # E_i per spec formula
    activation_weighted_error: float
    is_moe: bool = False
    expert_id: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


class SensitivityEngine:
    """Computes/estimates sensitivity metrics for tensors across candidate formats."""

    EPS = 1e-12

    # ------------------------------------------------ real weight measurement
    @classmethod
    def compute(
        cls,
        w: np.ndarray,
        w_hat: np.ndarray,
        tensor_name: str = "tensor",
        format_name: str = "quant",
        activations: Optional[np.ndarray] = None,
        is_moe: bool = False,
        expert_id: Optional[int] = None,
    ) -> SensitivityMetrics:
        """Measure real error between original and quantized weights."""
        diff = (w - w_hat).astype(np.float64)
        rms = float(np.sqrt(np.mean(diff ** 2)))
        max_err = float(np.max(np.abs(diff)))
        norm_orig = float(np.linalg.norm(w.astype(np.float64))) + cls.EPS
        l2_err = float(np.linalg.norm(diff)) / norm_orig

        # cosine similarity
        dot = float(np.sum(w.astype(np.float64) * w_hat.astype(np.float64)))
        norm_hat = float(np.linalg.norm(w_hat.astype(np.float64))) + cls.EPS
        cos = dot / (norm_orig * norm_hat)

        # SNR (dB)
        sig = np.mean(w.astype(np.float64) ** 2)
        noise = max(np.mean(diff ** 2), cls.EPS)
        snr = float(10.0 * np.log10((sig + cls.EPS) / noise))

        # activation-weighted: ||X(W - W_hat)||_F / ||XW||_F
        aw_err = l2_err
        if activations is not None:
            act = activations.astype(np.float64)
            xw = act @ w.astype(np.float64)
            xw_hat = act @ w_hat.astype(np.float64)
            aw_err = float(np.linalg.norm(xw - xw_hat)) / (float(np.linalg.norm(xw)) + cls.EPS)

        return SensitivityMetrics(
            tensor_name=tensor_name,
            format_name=format_name,
            rms_error=rms,
            max_error=max_err,
            cosine_sim=cos,
            snr_db=snr,
            normalized_l2=l2_err,
            activation_weighted_error=aw_err,
            is_moe=is_moe,
            expert_id=expert_id,
        )

    # ------------------------------------------------ analytic simulator (no weights needed)
    # Calibrated so that 16 bpw = 0 error, and Q1_0 gives visible but bounded error.
    # Uniform quantization variance: q^2 / 12  ->  SNR ≈ 6.02 * bpw + 1.76 dB
    FAMILY_IMPORTANCE: Dict[str, float] = {
        "lm_head": 1.5,
        "embed": 1.3,
        "norm": 2.0,           # norms must stay FP32
        "attn_out": 1.2,
        "mlp_down": 1.15,
        "attn_qkv": 1.0,
        "mlp_gate": 0.95,
        "mlp_up": 0.95,
        "moe_gate": 1.1,
        "moe_expert": 0.85,    # experts are numerous; individual sensitivity lower
    }

    @classmethod
    def estimate(
        cls,
        tensor: TensorMeta,
        fmt: QuantFormat,
        importance_override: Optional[float] = None,
    ) -> SensitivityMetrics:
        """Closed-form error estimate for offline planning when weights unavailable."""
        bpw = max(fmt.effective_bpw, 0.5)
        # Uniform quantizer noise theory
        snr = 6.02 * bpw - 4.5
        noise_var = 10 ** (-snr / 10.0)
        rms = math.sqrt(noise_var)
        l2 = math.sqrt(noise_var) * cls.FAMILY_IMPORTANCE.get(tensor.tensor_family, 1.0)
        cos = math.cos(math.atan(math.sqrt(noise_var)))
        importance = importance_override if importance_override is not None else cls.FAMILY_IMPORTANCE.get(tensor.tensor_family, 1.0)

        return SensitivityMetrics(
            tensor_name=tensor.name,
            format_name=fmt.name,
            rms_error=rms,
            max_error=rms * 3.5,
            cosine_sim=cos,
            snr_db=snr,
            normalized_l2=l2,
            activation_weighted_error=l2 * importance,
            is_moe=tensor.is_moe,
            expert_id=tensor.expert_id,
        )

    # ------------------------------------------------ export sensitivity maps
    @staticmethod
    def export(metrics: List[SensitivityMetrics], out_dir: str) -> None:
        import os
        os.makedirs(out_dir, exist_ok=True)
        # tensor-level
        with open(f"{out_dir}/tensor_sensitivity.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["tensor", "format", "rms", "max_err", "cosine", "snr_db", "norm_l2", "aw_err", "is_moe", "expert_id"])
            for m in metrics:
                w.writerow([m.tensor_name, m.format_name, f"{m.rms_error:.6f}", f"{m.max_error:.6f}",
                            f"{m.cosine_sim:.6f}", f"{m.snr_db:.2f}", f"{m.normalized_l2:.6f}",
                            f"{m.activation_weighted_error:.6f}", m.is_moe, m.expert_id if m.expert_id is not None else ""])

        # layer-level aggregation
        import re
        layers: Dict[int, List[SensitivityMetrics]] = {}
        for m in metrics:
            mm = re.search(r"layers\.(\d+)", m.tensor_name)
            lid = int(mm.group(1)) if mm else -1
            layers.setdefault(lid, []).append(m)

        with open(f"{out_dir}/layer_sensitivity.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["layer_id", "n_tensors", "mean_norm_l2", "max_norm_l2", "mean_aw_err", "mean_snr"])
            for lid in sorted(layers.keys()):
                ms = layers[lid]
                w.writerow([lid, len(ms),
                            f"{sum(m.normalized_l2 for m in ms)/len(ms):.6f}",
                            f"{max(m.normalized_l2 for m in ms):.6f}",
                            f"{sum(m.activation_weighted_error for m in ms)/len(ms):.6f}",
                            f"{sum(m.snr_db for m in ms)/len(ms):.2f}"])

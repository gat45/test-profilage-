"""Memory & I/O profiling (Profiler Axes 5, 6, 9).

Separates the three budgets defined in the spec:
  A. Storage budget  — on-disk footprint
  B. Memory budget   — RAM/VRAM available (working-set target)
  C. Movement budget — bytes transferred per generated token

Computes resident weights, KV cache, activations, expert cache, peak RSS,
cold/hot I/O, and enforces the "working-set ≤ limit" constraint (e.g. run a
>8 GiB model inside an 8 GiB budget).
"""

from __future__ import annotations
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from .profile_model import ModelMap
from .capability_db import DeviceProfile
from .quant_formats import FormatRegistry, quant_registry


@dataclass
class MemoryProfile:
    model_name: str
    device_name: str
    # Budgets
    storage_budget_gb: float          # physical storage used by checkpoint
    memory_budget_gb: float           # target RAM/VRAM ceiling
    # Components of working set
    resident_weights_gb: float        # weights kept permanently in memory
    kv_cache_gb: float                # context-dependent
    activations_gb: float             # peak activation memory
    expert_cache_gb: float            # MoE expert working cache
    overhead_gb: float                # framework & page-cache headroom
    working_set_gb: float             # sum of above
    peak_rss_gb: float
    exceeds_budget: bool
    context_tokens: int
    kv_precision: str                 # "fp16", "int8", "int4"
    # I/O axis
    cold_bytes_per_token: float       # first-time loaded weights per token
    hot_bytes_per_token: float        # already-cached activations/state
    io_per_token_mb: float            # storage reads per token generated
    tokens_until_cache_hot: int       # warmup before steady state
    # KV breakdown
    kv_bytes_per_token_total: float

    def to_dict(self) -> dict:
        return asdict(self)


class MemoryProfiler:
    """Profiles memory budgets and I/O traffic for a model+device+plan configuration."""

    def __init__(self, registry: Optional[FormatRegistry] = None) -> None:
        self.registry = registry or quant_registry

    # ------------------------------------------------ KV cache (Axe 9)
    @staticmethod
    def kv_cache_gb(model: ModelMap, context_tokens: int, precision: str = "fp16") -> Tuple[float, float]:
        """Returns (kv_cache_gb, bytes_per_token).

        Formula: 2 * n_layers * n_kv_heads * head_dim * context * bytes_per_value
        """
        bytes_per_value = {"fp16": 2, "bf16": 2, "int8": 1, "int4": 0.5}[precision.lower()]
        head_dim = model.hidden_size // model.num_attention_heads
        per_token = 2 * model.num_hidden_layers * model.num_key_value_heads * head_dim * bytes_per_value
        total = per_token * context_tokens
        return total / (1024 ** 3), per_token

    # ------------------------------------------------ activations (peak)
    @staticmethod
    def activations_gb(model: ModelMap, context_tokens: int, batch_size: int = 1) -> float:
        """Peak activation memory during prefill (roughly batch * context * hidden * 2 * n_layers * 0.1)."""
        peak_bytes = batch_size * max(context_tokens, 1) * model.hidden_size * 2 * min(model.num_hidden_layers, 8) * 0.25
        return peak_bytes / (1024 ** 3)

    # ------------------------------------------------ working set & 3 budgets
    def profile(
        self,
        model: ModelMap,
        device: DeviceProfile,
        quant_plan: Optional[Dict[str, str]] = None,      # tensor_name -> format_name
        resident_tensors: Optional[List[str]] = None,     # names kept resident
        context_tokens: int = 4096,
        kv_precision: str = "fp16",
        expert_cache_gb: float = 0.5,
        memory_budget_gb: Optional[float] = None,
        streaming_enabled: bool = True,
    ) -> MemoryProfile:
        """Compute the full memory profile including the 3 budgets."""

        # (1) Storage budget A — checkpoint size with chosen formats
        storage_bytes = 0
        for t in model.tensors:
            fmt_name = (quant_plan or {}).get(t.name, t.dtype)
            try:
                fmt = self.registry.get(fmt_name)
                storage_bytes += fmt.quantized_bytes(t.n_params)
            except KeyError:
                storage_bytes += t.size_bytes
        storage_gb = storage_bytes / (1024 ** 3)

        # (2) Resident weights — default: dense trunk resident, MoE experts streamed
        if resident_tensors is None:
            if streaming_enabled and model.is_moe:
                resident_names = {t.name for t in model.dense_tensors()}
            else:
                resident_names = {t.name for t in model.tensors}
        else:
            resident_names = set(resident_tensors)

        resident_bytes = 0
        for t in model.tensors:
            if t.name in resident_names:
                fmt_name = (quant_plan or {}).get(t.name, t.dtype)
                try:
                    fmt = self.registry.get(fmt_name)
                    resident_bytes += fmt.quantized_bytes(t.n_params)
                except KeyError:
                    resident_bytes += t.size_bytes
        resident_gb = resident_bytes / (1024 ** 3)

        # (3) KV cache (Axe 9)
        kv_gb, kv_per_tok = self.kv_cache_gb(model, context_tokens, kv_precision)

        # (4) activations + overhead
        act_gb = self.activations_gb(model, context_tokens)
        overhead_gb = max(0.5, 0.08 * (device.total_addressable_gb or 8.0))

        # (5) working set
        expert_cache_gb = expert_cache_gb if model.is_moe else 0.0
        ws_gb = resident_gb + kv_gb + act_gb + expert_cache_gb + overhead_gb
        peak_rss_gb = ws_gb * 1.05

        budget_gb = memory_budget_gb or device.total_addressable_gb or 8.0

        # (6) Movement budget C — cold/hot bytes per token
        # For streamed models: expert bytes demand per token = top_k * expert_size (if cache miss)
        moe_total_bytes = sum(t.size_bytes for t in model.expert_tensors())
        bytes_per_expert = moe_total_bytes / max(model.num_experts, 1)
        cold_bytes_tok = (model.top_k * bytes_per_expert * 0.3) if model.is_moe and streaming_enabled else 0.0
        hot_bytes_tok = kv_per_tok
        io_mb = cold_bytes_tok / (1024 ** 2)

        return MemoryProfile(
            model_name=model.name,
            device_name=device.name,
            storage_budget_gb=round(storage_gb, 2),
            memory_budget_gb=round(budget_gb, 2),
            resident_weights_gb=round(resident_gb, 2),
            kv_cache_gb=round(kv_gb, 2),
            activations_gb=round(act_gb, 2),
            expert_cache_gb=round(expert_cache_gb, 2),
            overhead_gb=round(overhead_gb, 2),
            working_set_gb=round(ws_gb, 2),
            peak_rss_gb=round(peak_rss_gb, 2),
            exceeds_budget=ws_gb > budget_gb,
            context_tokens=context_tokens,
            kv_precision=kv_precision,
            cold_bytes_per_token=cold_bytes_tok,
            hot_bytes_per_token=hot_bytes_tok,
            io_per_token_mb=round(io_mb, 2),
            tokens_until_cache_hot=min(max(model.num_experts, 1), 64),
            kv_bytes_per_token_total=kv_per_tok,
        )

    def print_budget_table(self, profile: MemoryProfile) -> str:
        s = profile
        return (
            f"=== 3-Budget Report : {s.model_name} on {s.device_name} ===\n"
            f"  A. Storage budget (on disk)  : {s.storage_budget_gb:.1f} GB\n"
            f"  B. Memory budget (ceiling)   : {s.memory_budget_gb:.1f} GB\n"
            f"     ├─ resident weights       : {s.resident_weights_gb:.2f} GB\n"
            f"     ├─ KV cache (ctx={s.context_tokens})  : {s.kv_cache_gb:.2f} GB ({s.kv_precision})\n"
            f"     ├─ activations (peak)     : {s.activations_gb:.2f} GB\n"
            f"     ├─ expert cache           : {s.expert_cache_gb:.2f} GB\n"
            f"     └─ overhead               : {s.overhead_gb:.2f} GB\n"
            f"     working set               : {s.working_set_gb:.2f} GB "
            f"({'EXCEEDS' if s.exceeds_budget else 'OK'})\n"
            f"  C. Movement budget           : {s.io_per_token_mb:.2f} MB / token "
            f"(cold={s.cold_bytes_per_token/1e6:.1f}MB, hot={s.hot_bytes_per_token/1e3:.1f}KB)"
        )

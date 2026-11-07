"""Structural model profiling (Profiler Axe 1 — Structural mapping).

Reconstructs the real organization of weights and operations (layers, tensors,
families, MoE experts) from GGUF metadata, Safetensors headers, or Hugging Face
configs, and exports model_map.json / layer_map.csv / tensor_map.csv / architecture_report.json.
"""

from __future__ import annotations
import csv
import json
import math
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class TensorMeta:
    name: str
    shape: List[int]
    dtype: str
    n_params: int
    size_bytes: int
    layer_id: int                    # -1 for non-layer tensors
    tensor_family: str               # "embed","lm_head","attn_qkv","attn_out","mlp_up","mlp_down","mlp_gate","norm","moe_expert","moe_gate"
    is_moe: bool = False
    expert_id: Optional[int] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelMap:
    """Structural cartography of a model."""
    name: str
    model_type: str                  # e.g. "llama", "mixtral", "kimi_k3", "bonsai"
    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    vocab_size: int
    num_experts: int = 0             # 0 for dense
    top_k: int = 0
    tensors: List[TensorMeta] = field(default_factory=list)
    source: str = "synthetic"        # "gguf", "safetensors", "hf_config", "synthetic"
    notes: str = ""

    # ---------------------------------------------------------------- aggregate stats
    @property
    def total_params(self) -> int:
        return sum(t.n_params for t in self.tensors)

    @property
    def is_moe(self) -> bool:
        return self.num_experts > 0

    @property
    def n_params_per_token_decode(self) -> int:
        """Parameters active during a single decode step (top_k experts + dense trunk)."""
        if not self.is_moe:
            return self.total_params
        dense = sum(t.n_params for t in self.tensors if not t.is_moe)
        moe_total = sum(t.n_params for t in self.tensors if t.is_moe)
        per_expert = moe_total / max(self.num_experts, 1)
        return int(dense + self.top_k * per_expert)

    @property
    def raw_size_gb_bf16(self) -> float:
        return self.total_params * 2 / (1024 ** 3)

    # ---------------------------------------------------------------- construction
    @classmethod
    def from_hf_config(cls, cfg: dict, name: str = "hf_model") -> "ModelMap":
        """Synthesize a complete tensor map from a HuggingFace-style config.json."""
        hidden = cfg.get("hidden_size", 4096)
        layers = cfg.get("num_hidden_layers", 32)
        heads = cfg.get("num_attention_heads", 32)
        kv_heads = cfg.get("num_key_value_heads", heads)
        inter = cfg.get("intermediate_size", int(hidden * 8 / 3))
        vocab = cfg.get("vocab_size", 32000)
        n_experts = cfg.get("num_experts", cfg.get("num_local_experts", 0))
        top_k = cfg.get("top_k", cfg.get("num_experts_per_tok", 0))
        head_dim = cfg.get("head_dim", hidden // heads)
        kv_dim = kv_heads * head_dim

        mtype = cfg.get("model_type", "llama")
        model = cls(
            name=name,
            model_type=mtype,
            hidden_size=hidden,
            num_hidden_layers=layers,
            num_attention_heads=heads,
            num_key_value_heads=kv_heads,
            intermediate_size=inter,
            vocab_size=vocab,
            num_experts=n_experts,
            top_k=top_k,
            source="hf_config",
        )
        t: List[TensorMeta] = []

        def add(nm, shape, dtype, layer, family, is_moe=False, expert=None):
            n = math.prod(shape)
            bytes_ = n * 2 if dtype in ("BF16", "FP16") else n * 4
            t.append(TensorMeta(nm, shape, dtype, n, bytes_, layer, family, is_moe, expert))

        # embed + lm_head
        add("model.embed_tokens.weight", [vocab, hidden], "BF16", -1, "embed")
        for i in range(layers):
            pfx = f"model.layers.{i}"
            # attention
            add(f"{pfx}.self_attn.q_proj.weight", [heads * head_dim, hidden], "BF16", i, "attn_qkv")
            add(f"{pfx}.self_attn.k_proj.weight", [kv_dim, hidden], "BF16", i, "attn_qkv")
            add(f"{pfx}.self_attn.v_proj.weight", [kv_dim, hidden], "BF16", i, "attn_qkv")
            add(f"{pfx}.self_attn.o_proj.weight", [hidden, heads * head_dim], "BF16", i, "attn_out")
            add(f"{pfx}.input_layernorm.weight", [hidden], "FP32", i, "norm")
            add(f"{pfx}.post_attention_layernorm.weight", [hidden], "FP32", i, "norm")
            # MLP — dense or MoE
            if n_experts == 0:
                add(f"{pfx}.mlp.gate_proj.weight", [inter, hidden], "BF16", i, "mlp_gate")
                add(f"{pfx}.mlp.up_proj.weight", [inter, hidden], "BF16", i, "mlp_up")
                add(f"{pfx}.mlp.down_proj.weight", [hidden, inter], "BF16", i, "mlp_down")
            else:
                add(f"{pfx}.block_sparse_moe.gate.weight", [n_experts, hidden], "BF16", i, "moe_gate")
                for e in range(n_experts):
                    add(f"{pfx}.block_sparse_moe.experts.{e}.gate_proj.weight",
                        [inter, hidden], "BF16", i, "moe_expert", True, e)
                    add(f"{pfx}.block_sparse_moe.experts.{e}.up_proj.weight",
                        [inter, hidden], "BF16", i, "moe_expert", True, e)
                    add(f"{pfx}.block_sparse_moe.experts.{e}.down_proj.weight",
                        [hidden, inter], "BF16", i, "moe_expert", True, e)
        add("model.norm.weight", [hidden], "FP32", layers, "norm")
        add("lm_head.weight", [vocab, hidden], "BF16", layers + 1, "lm_head")

        model.tensors = t
        return model

    # ---------------------------------------------------------------- presets matching documented models
    @classmethod
    def qwen35_9b(cls) -> "ModelMap":
        """Qwen3.5-9B-like dense model (~9B params). Used in GTX 1080 validation."""
        return cls.from_hf_config({
            "model_type": "qwen",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 12288,
            "vocab_size": 152064,
        }, name="Qwen3.5-9B")

    @classmethod
    def bonsai_27b(cls) -> "ModelMap":
        """Bonsai-27B-like dense model (~27B params, Q1_0 format reference)."""
        return cls.from_hf_config({
            "model_type": "bonsai",
            "hidden_size": 4608,
            "num_hidden_layers": 46,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 36864,
            "vocab_size": 256000,
        }, name="Bonsai-27B")

    @classmethod
    def kimi_k3(cls) -> "ModelMap":
        """Kimi K3-like MoE: dense BF16 trunk + 128 MXFP4 experts, Top-16 routing."""
        m = cls.from_hf_config({
            "model_type": "deepseek_moe",
            "hidden_size": 4096,
            "num_hidden_layers": 60,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 1536,          # per-expert intermediate
            "vocab_size": 102400,
            "num_experts": 128,
            "top_k": 16,
        }, name="Kimi-K3")
        # expert tensors start in MXFP4 format as per spec
        for t in m.tensors:
            if t.is_moe:
                t.dtype = "MXFP4"
                t.size_bytes = int(math.ceil(t.n_params * 4.0 / 8.0))
        m.notes = "Dense trunk BF16; 128 experts in MXFP4; Top-16 router."
        return m

    @classmethod
    def marco_moe(cls) -> "ModelMap":
        """Small Marco-style MoE (~2-3B) used in mobile validation numbers."""
        return cls.from_hf_config({
            "model_type": "moe_small",
            "hidden_size": 2048,
            "num_hidden_layers": 16,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "intermediate_size": 2048,
            "vocab_size": 32000,
            "num_experts": 8,
            "top_k": 2,
        }, name="Marco-MoE")

    # ---------------------------------------------------------------- filtering
    def tensors_by_family(self, family: str) -> List[TensorMeta]:
        return [t for t in self.tensors if t.tensor_family == family]

    def tensors_by_layer(self, layer_id: int) -> List[TensorMeta]:
        return [t for t in self.tensors if t.layer_id == layer_id]

    def expert_tensors(self) -> List[TensorMeta]:
        return [t for t in self.tensors if t.is_moe]

    def dense_tensors(self) -> List[TensorMeta]:
        return [t for t in self.tensors if not t.is_moe]

    # ---------------------------------------------------------------- exports
    def export(self, out_dir: str) -> None:
        """Writes model_map.json, layer_map.csv, tensor_map.csv, architecture_report.json."""
        import os
        os.makedirs(out_dir, exist_ok=True)

        with open(f"{out_dir}/model_map.json", "w") as f:
            json.dump({
                "name": self.name,
                "model_type": self.model_type,
                "total_params": self.total_params,
                "num_hidden_layers": self.num_hidden_layers,
                "hidden_size": self.hidden_size,
                "num_experts": self.num_experts,
                "top_k": self.top_k,
                "is_moe": self.is_moe,
                "source": self.source,
                "notes": self.notes,
                "tensors": [t.to_dict() for t in self.tensors],
            }, f, indent=1)

        with open(f"{out_dir}/tensor_map.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "shape", "dtype", "n_params", "size_bytes",
                             "layer_id", "family", "is_moe", "expert_id"])
            for t in self.tensors:
                writer.writerow([t.name, "x".join(map(str, t.shape)), t.dtype,
                                 t.n_params, t.size_bytes, t.layer_id, t.tensor_family,
                                 t.is_moe, t.expert_id if t.expert_id is not None else ""])

        # per-layer aggregate
        layers: Dict[int, dict] = {}
        for t in self.tensors:
            l = layers.setdefault(t.layer_id, {"params": 0, "bytes": 0, "families": set()})
            l["params"] += t.n_params
            l["bytes"] += t.size_bytes
            l["families"].add(t.tensor_family)
        with open(f"{out_dir}/layer_map.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["layer_id", "params", "bytes", "families"])
            for lid in sorted(layers.keys()):
                writer.writerow([lid, layers[lid]["params"], layers[lid]["bytes"],
                                 "|".join(sorted(layers[lid]["families"]))])

        with open(f"{out_dir}/architecture_report.json", "w") as f:
            json.dump(self.architecture_report(), f, indent=2)

    def architecture_report(self) -> dict:
        families: Dict[str, dict] = {}
        for t in self.tensors:
            fam = families.setdefault(t.tensor_family, {"count": 0, "params": 0, "bytes": 0})
            fam["count"] += 1
            fam["params"] += t.n_params
            fam["bytes"] += t.size_bytes
        return {
            "name": self.name,
            "total_params": self.total_params,
            "total_params_B": round(self.total_params / 1e9, 2),
            "layers": self.num_hidden_layers,
            "hidden_size": self.hidden_size,
            "is_moe": self.is_moe,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "active_params_per_token_B": round(self.n_params_per_token_decode / 1e9, 2),
            "raw_size_gb_bf16": round(self.raw_size_gb_bf16, 2),
            "n_tensors": len(self.tensors),
            "family_breakdown": {
                k: {**v, "share_pct": round(100 * v["params"] / max(self.total_params, 1), 1)}
                for k, v in families.items()
            },
        }

    def summary(self) -> str:
        r = self.architecture_report()
        lines = [
            f"== {self.name} ==",
            f"  params: {r['total_params_B']}B  |  layers: {self.num_hidden_layers}  |  hidden: {self.hidden_size}",
            f"  moe: {r['is_moe']} (experts={self.num_experts}, top_k={self.top_k})  |  active/tok: {r['active_params_per_token_B']}B",
            f"  raw BF16: {r['raw_size_gb_bf16']} GB  |  tensors: {r['n_tensors']}",
            "  families: " + ", ".join(
                f"{k}={v['share_pct']}%" for k, v in r["family_breakdown"].items()),
        ]
        return "\n".join(lines)

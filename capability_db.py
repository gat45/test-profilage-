"""Capability DB — Hardware profiling database for Z2/D2 Planner.

Stores device profiles with compute, memory, and transfer parameters used by
the cost model and planner to choose placements and predict performance.
"""

from __future__ import annotations
import json
import platform
import shutil
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional


@dataclass
class DeviceProfile:
    """Hardware capability profile for one target device."""
    name: str
    label: str
    device_type: str                  # "gpu", "cpu", "npu", "soc"
    # Memory
    ram_gb: float = 0.0               # system RAM
    vram_gb: float = 0.0              # dedicated VRAM
    vtcm_mb: float = 0.0              # on-chip NPU memory (e.g. Hexagon VTCM)
    memory_bw_gbps: float = 25.0      # main memory bandwidth
    storage_read_mbps: float = 500.0  # cold read throughput from disk
    # Compute
    fp16_tflops: float = 1.0          # FP16 half-precision throughput
    int8_tops: float = 2.0            # INT8 throughput
    bf16_supported: bool = False
    mxfp4_supported: bool = False
    fp8_supported: bool = False
    # Runtimes
    has_cuda: bool = False
    has_opencl: bool = False
    has_htp: bool = False             # Qualcomm HTP / QAIRT
    htp_generation: Optional[str] = None  # e.g. "v81"
    runtime_notes: str = ""
    # Transfer costs (host <-> accelerator)
    transfer_bw_gbps: float = 16.0    # PCIe / SoC bus bandwidth
    rpc_overhead_ms: float = 0.0      # RPC invocation overhead (HTP, DSP)
    dma_available: bool = False
    # Empirical validation flags
    validated: bool = False           # has been empirically tested
    cpu_fallback_risk: float = 0.0    # empirical % MUL_MAT falling back to CPU (0-1)

    @property
    def total_addressable_gb(self) -> float:
        """Total memory available to the device (VRAM if dedicated GPU, else RAM)."""
        return self.vram_gb if self.vram_gb > 0 else self.ram_gb

    @property
    def best_tflops(self) -> float:
        """Rough aggregate compute figure in TFLOPS-equivalent."""
        return max(self.fp16_tflops, self.int8_tops / 2.0)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> DeviceProfile:
        return cls(**data)


# ---------------------------------------------------------------------------
# Built-in profiles, including all devices referenced in the project document
# ---------------------------------------------------------------------------
BUILTIN_PROFILES: Dict[str, DeviceProfile] = {
    # Validated reference: GTX 1080 8GB (Qwen3.5-9B documented validation)
    "gtx_1080_8gb": DeviceProfile(
        name="gtx_1080_8gb",
        label="NVIDIA GeForce GTX 1080 (8 GB GDDR5)",
        device_type="gpu",
        vram_gb=8.0,
        ram_gb=32.0,
        memory_bw_gbps=320.0,
        storage_read_mbps=550.0,
        fp16_tflops=8.87,
        int8_tops=17.7,
        has_cuda=True,
        transfer_bw_gbps=16.0,          # PCIe 3.0 x16
        validated=True,
        runtime_notes="Reference hardware for mixed-precision validation on Qwen3.5-9B (PPL 9.78 / 33.42 tok/s).",
    ),

    # Validated compact-format reference: GTX 1080 Ti (Bonsai-27B Q1_0)
    "gtx_1080_ti_11gb": DeviceProfile(
        name="gtx_1080_ti_11gb",
        label="NVIDIA GeForce GTX 1080 Ti (11 GB GDDR5X)",
        device_type="gpu",
        vram_gb=11.0,
        ram_gb=32.0,
        memory_bw_gbps=484.0,
        storage_read_mbps=550.0,
        fp16_tflops=11.34,
        int8_tops=22.7,
        has_cuda=True,
        transfer_bw_gbps=16.0,
        validated=True,
        runtime_notes="Q1_0 format reference (1.125 bpw) on Bonsai-27B (~28.3 tok/s reported).",
    ),

    # Simulation platform 1: Ryzen 5 5600 + RTX 3060 12GB
    "ryzen5_5600_32gb": DeviceProfile(
        name="ryzen5_5600_32gb",
        label="AMD Ryzen 5 5600 (32 GB DDR4)",
        device_type="cpu",
        ram_gb=32.0,
        memory_bw_gbps=25.6,
        storage_read_mbps=3500.0,       # NVMe
        fp16_tflops=0.57,
        int8_tops=1.1,
        runtime_notes="Simulation station CPU — AVX2/FMA, NVMe storage.",
    ),
    "rtx_3060_12gb": DeviceProfile(
        name="rtx_3060_12gb",
        label="NVIDIA GeForce RTX 3060 (12 GB GDDR6)",
        device_type="gpu",
        vram_gb=12.0,
        ram_gb=32.0,
        memory_bw_gbps=360.0,
        storage_read_mbps=3500.0,
        fp16_tflops=12.7,
        int8_tops=100.0,                # 2nd-gen tensor cores
        bf16_supported=True,
        has_cuda=True,
        transfer_bw_gbps=32.0,          # PCIe 4.0 x16
        runtime_notes="Simulation platform 1 GPU — Ampere architecture.",
    ),

    # Simulation platform 2: Ryzen AI 9 HX365 + RTX 5070 8GB
    "ryzen_ai9_hx365_32gb": DeviceProfile(
        name="ryzen_ai9_hx365_32gb",
        label="AMD Ryzen AI 9 HX365 (32 GB LPDDR5X + XDNA2 NPU)",
        device_type="cpu",
        ram_gb=32.0,
        memory_bw_gbps=120.0,
        storage_read_mbps=5000.0,
        fp16_tflops=1.2,
        int8_tops=50.0,                 # XDNA2 NPU
        bf16_supported=True,
        fp8_supported=True,
        runtime_notes="Modern station — integrated NPU (50 TOPS) + fast LPDDR5X.",
    ),
    "rtx_5070_8gb": DeviceProfile(
        name="rtx_5070_8gb",
        label="NVIDIA GeForce RTX 5070 (8 GB GDDR7)",
        device_type="gpu",
        vram_gb=8.0,
        ram_gb=32.0,
        memory_bw_gbps=504.0,
        storage_read_mbps=5000.0,
        fp16_tflops=100.0,
        int8_tops=400.0,
        bf16_supported=True,
        fp8_supported=True,
        has_cuda=True,
        transfer_bw_gbps=32.0,
        runtime_notes="Simulation platform 2 GPU — Blackwell generation, memory-constrained.",
    ),

    # Simulation platform 3: OnePlus 15 (Snapdragon 8 Elite / HTP v81 / Adreno 830)
    "oneplus_15_htp_v81": DeviceProfile(
        name="oneplus_15_htp_v81",
        label="OnePlus 15 — Qualcomm Snapdragon 8 Elite (HTP v81 + Adreno 830)",
        device_type="soc",
        ram_gb=16.0,
        memory_bw_gbps=85.0,            # LPDDR5X
        storage_read_mbps=4000.0,       # UFS 4.0
        vtcm_mb=8.0,                    # Hexagon VTCM
        fp16_tflops=12.0,               # Adreno 830
        int8_tops=80.0,                 # HTP v81 NPU
        bf16_supported=False,           # HTP favors INT4/INT8/FP8
        fp8_supported=True,
        mxfp4_supported=True,           # MXFP4 target for Kimi K3 experts
        has_opencl=True,
        has_htp=True,
        htp_generation="v81",
        dma_available=True,
        transfer_bw_gbps=85.0,          # unified SoC bus
        rpc_overhead_ms=0.3,
        validated=True,
        cpu_fallback_risk=0.15,         # calibrated risk; Bonsai stress test saw 99.7% CPU fallback
        runtime_notes=(
            "Validated against reported mobile numbers (Marco MoE: 31.23 tok/s OpenCL / "
            "23.6 tok/s HTP; 9B: ~10 tok/s; 1.7B: ~38 tok/s). Beware kernel compatibility: "
            "Bonsai-27B MUL_MAT fallback was 99.7%."
        ),
    ),

    # Adreno 830 standalone GPU slice
    "adreno_830": DeviceProfile(
        name="adreno_830",
        label="Qualcomm Adreno 830 (OpenCL)",
        device_type="gpu",
        ram_gb=16.0,
        memory_bw_gbps=85.0,
        storage_read_mbps=4000.0,
        fp16_tflops=12.0,
        int8_tops=24.0,
        has_opencl=True,
        transfer_bw_gbps=85.0,
        validated=True,
        runtime_notes="Mobile GPU slice — OpenCL GEMM kernels; faster than HTP on some MoE models.",
    ),

    # Data-center reference (for prediction calibration)
    "a100_80gb": DeviceProfile(
        name="a100_80gb",
        label="NVIDIA A100 SXM4 (80 GB HBM2e)",
        device_type="gpu",
        vram_gb=80.0,
        ram_gb=256.0,
        memory_bw_gbps=2039.0,
        storage_read_mbps=7000.0,
        fp16_tflops=312.0,
        int8_tops=624.0,
        bf16_supported=True,
        has_cuda=True,
        transfer_bw_gbps=64.0,
        runtime_notes="Server reference profile.",
    ),
}


class CapabilityDB:
    """Registry of hardware profiles with discovery and persistence."""

    def __init__(self) -> None:
        self._profiles: Dict[str, DeviceProfile] = dict(BUILTIN_PROFILES)

    def register(self, profile: DeviceProfile) -> None:
        self._profiles[profile.name] = profile

    def get(self, name: str) -> DeviceProfile:
        if name not in self._profiles:
            raise KeyError(
                f"Unknown device '{name}'. Available: {list(self._profiles.keys())}"
            )
        return self._profiles[name]

    def list_devices(self) -> List[DeviceProfile]:
        return list(self._profiles.values())

    def names(self) -> List[str]:
        return list(self._profiles.keys())

    def discover_host(self) -> DeviceProfile:
        """Best-effort probe of the current machine (Linux only)."""
        ram_gb = 16.0
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        ram_gb = round(int(line.split()[1]) / 1024 / 1024, 1)
                        break
        except OSError:
            pass

        has_cuda = shutil.which("nvidia-smi") is not None
        vram_gb = 0.0
        label = f"{platform.processor() or platform.machine()} ({ram_gb} GB RAM)"
        if has_cuda:
            import subprocess
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                    text=True, timeout=5,
                ).strip().splitlines()[0]
                name, mem = out.split(",")
                vram_gb = round(float(mem) / 1024, 1)
                label = f"{name.strip()} ({vram_gb} GB VRAM) + host {ram_gb} GB RAM"
            except Exception:
                pass

        profile = DeviceProfile(
            name="host",
            label=label,
            device_type="gpu" if has_cuda else "cpu",
            ram_gb=ram_gb,
            vram_gb=vram_gb,
            has_cuda=has_cuda,
            memory_bw_gbps=300.0 if has_cuda else 25.0,
            fp16_tflops=10.0 if has_cuda else 0.5,
            int8_tops=20.0 if has_cuda else 1.0,
            storage_read_mbps=1500.0,
            runtime_notes="Auto-discovered host profile.",
        )
        self.register(profile)
        return profile

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({k: v.to_dict() for k, v in self._profiles.items()}, f, indent=2)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            self._profiles[k] = DeviceProfile.from_dict(v)

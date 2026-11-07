import tempfile
import numpy as np
import pytest
from adaptive_quant.profile_model import ModelMap
from adaptive_quant.sensitivity_engine import SensitivityEngine
from adaptive_quant.quantization_engine import QuantizationEngine
from adaptive_quant.profiler import MemoryProfiler


def test_presets():
    q = ModelMap.qwen35_9b()
    assert 6e9 < q.total_params < 12e9
    assert q.is_moe is False
    assert len(q.tensors) > 200

    b = ModelMap.bonsai_27b()
    assert 20e9 < b.total_params < 35e9

    k = ModelMap.kimi_k3()
    assert k.is_moe is True
    assert k.num_experts == 128
    assert k.top_k == 16
    assert k.n_params_per_token_decode < k.total_params


def test_model_export(tmp_path):
    m = ModelMap.qwen35_9b()
    m.export(str(tmp_path))
    assert (tmp_path / "model_map.json").exists()
    assert (tmp_path / "layer_map.csv").exists()
    assert (tmp_path / "tensor_map.csv").exists()
    assert (tmp_path / "architecture_report.json").exists()


def test_sensitivity_metrics():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((128, 128)).astype(np.float32)
    qe = QuantizationEngine()
    w_hat = qe.quant_dequant_int(w, bits=4, block_size=32)
    sm = SensitivityEngine.compute(w, w_hat, "test_tensor", "INT4")
    assert 0 < sm.rms_error < 0.2
    assert sm.snr_db > 10
    assert 0.9 < sm.cosine_sim <= 1.0


def test_numeric_kernels_reduce_error_with_more_bits():
    rng = np.random.default_rng(1)
    w = rng.standard_normal((256, 256)).astype(np.float32)
    qe = QuantizationEngine()
    e8 = float(np.mean((w - qe.quant_dequant_int(w, 8)) ** 2))
    e4 = float(np.mean((w - qe.quant_dequant_int(w, 4)) ** 2))
    e2 = float(np.mean((w - qe.quant_dequant_int(w, 2)) ** 2))
    assert e8 < e4 < e2


def test_awq_calibration():
    rng = np.random.default_rng(2)
    w = rng.standard_normal((64, 64)).astype(np.float32)
    x = rng.standard_normal((8, 64)).astype(np.float32)
    qe = QuantizationEngine()
    mixed, info = qe.calibrate_awq(w, x, base_bits=4, salient_fraction=0.02)
    plain = qe.quant_dequant_int(w, 4)
    assert np.mean((w - mixed) ** 2) < np.mean((w - plain) ** 2)


def test_kv_cache_scaling():
    m = ModelMap.qwen35_9b()
    gb_4k, _ = MemoryProfiler.kv_cache_gb(m, 4096, "fp16")
    gb_8k, _ = MemoryProfiler.kv_cache_gb(m, 8192, "fp16")
    gb_int4, _ = MemoryProfiler.kv_cache_gb(m, 4096, "int4")
    assert abs(gb_8k - 2 * gb_4k) < 1e-4
    assert abs(gb_int4 - 0.25 * gb_4k) < 1e-4

import pytest
from adaptive_quant.capability_db import CapabilityDB, DeviceProfile
from adaptive_quant.quant_formats import quant_registry, QuantFormat


def test_builtin_devices_present():
    db = CapabilityDB()
    for name in ("gtx_1080_8gb", "gtx_1080_ti_11gb", "ryzen5_5600_32gb", "rtx_3060_12gb",
                 "ryzen_ai9_hx365_32gb", "rtx_5070_8gb", "oneplus_15_htp_v81", "a100_80gb"):
        dev = db.get(name)
        assert dev.total_addressable_gb > 0


def test_device_discovery():
    db = CapabilityDB()
    host = db.discover_host()
    assert host.ram_gb > 0


def test_quant_formats():
    assert quant_registry.get("Q4_0").effective_bpw == 4.5
    assert quant_registry.get("Q1_0").effective_bpw == 1.125
    assert quant_registry.get("MXFP4").effective_bpw == 4.0
    for f in ("FP16", "BF16", "INT8", "INT4", "Q8_0", "Q6_K", "Q2_K", "IQ4_XS"):
        fmt = quant_registry.get(f)
        assert fmt.quantized_bytes(1_000_000) > 0


def test_format_size_order():
    n = 10_000_000
    sz_fp32 = quant_registry.get("FP32").quantized_bytes(n)
    sz_q4 = quant_registry.get("Q4_0").quantized_bytes(n)
    sz_q1 = quant_registry.get("Q1_0").quantized_bytes(n)
    assert sz_fp32 > sz_q4 > sz_q1

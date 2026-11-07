# LLM Adaptive Quantization & Hardware Profiling Framework

> **Reference implementation** of the complete architecture specified in
> _"Architecture complète du projet de quantification adaptative et de profilage matériel"_
> (Profiler V3/V5 · Quantification Algo · Z2/D2 Planner · Kimi K3 application).

[![tests](https://img.shields.io/badge/pytest-20%2F20%20passed-brightgreen)](#) Python 3.12 · 14 modules · 100% pure Python (+ numpy, matplotlib, tabulate)

---

## 1. Overview

This framework finds the **optimal numerical representation and execution
strategy** for large language models that exceed available memory. Rather than
blindly reducing bits per weight, it jointly optimizes:

- numerical precision, checkpoint size, output quality
- resident memory, transfer bandwidth, accelerator placement
- dequantization cost, dense vs MoE behavior, latency & tok/s

**Central cost model** (Profiler Axe 12):
```
T_token = T_io + T_compute + T_dequant + T_sync + T_other
```

## 2. Project Layout

```text
quant_project/
├── doc/                        # Technical report (Typst source + PDF)
│   ├── main.typ
│   ├── report-theme.typ
│   ├── helpers.typ
│   └── main.pdf               # 18-page compiled reference document
├── figures/                    # Diagrams (D2/Mermaid sources + PNG renders)
├── src/adaptive_quant/         # Python framework (14 modules)
├── tests/                      # pytest test suite (20 tests)
├── examples/                   # Reproducible demos
└── out/                        # Exported plans & decision logs
```

## 3. Quick Start

```bash
cd quant_project
export PYTHONPATH=src

# List devices and formats
python3 -m adaptive_quant.cli devices
python3 -m adaptive_quant.cli formats

# End-to-end plan for Kimi K3 on OnePlus 15 (exports quantization_plan.json)
python3 -m adaptive_quant.cli plan --model kimi_k3 --device oneplus_15_htp_v81 --out out/

# 3-budget memory profile
python3 -m adaptive_quant.cli profile --model kimi_k3 --device oneplus_15_htp_v81

# Offline performance prediction
python3 -m adaptive_quant.cli predict --model kimi_k3 --device oneplus_15_htp_v81 --v5

# MoE expert cache study (LRU vs LFU vs adaptive)
python3 -m adaptive_quant.cli simulate-cache --experts 128 --topk 16 --tokens 200 --cap 32

# L1–L3 validation of the GTX 1080 plan
python3 -m adaptive_quant.cli validate --model qwen35_9b --device gtx_1080_8gb

# Reproduce the documented GTX 1080 validation
python3 examples/demo_gtx1080_validation.py

# Simulate Kimi K3 on the 3 target platforms
python3 examples/demo_kimi_k3.py
```

## 4. Module Map (Section 7 of the spec)

| Module | Responsibility | Doc axe |
|---|---|---|
| `capability_db.py` | Hardware profiles (GTX 1080/Ti, RTX 3060/5070, Ryzen, HTP v81, A100) + host discovery | — |
| `quant_formats.py` | Format registry (BF16, FP16, INT8/4/2, Q8_0–Q1_0, MXFP4, E8M0, IQ) with physical cost models | Profiler Axe 2 |
| `profile_model.py` | Structural cartography; exports `model_map.json`, `layer_map.csv`, `tensor_map.csv` | Profiler Axe 1 |
| `profiler.py` | 3 budgets (storage/memory/movement), KV cache, working-set checks | Axes 5, 6, 9 |
| `cost_model.py` | Hardware-aware latency breakdown + overlapped decode/prefill tok/s | Axes 10, 12 |
| `predict_from_hf.py` | Offline prediction pass (storage, working set, tok/s, confidence intervals) | Axe 12 |
| `predict_from_hf_v5.py` | V5 prediction: storage ceilings, calibrated intervals, MTP projection | Axe 12 (V5) |
| `parse_hexagon.py` | Qualcomm HTP/QAIRT trace parser; detects CPU fallbacks on `MUL_MAT` | Axe 7 |
| `sensitivity_engine.py` | RMS, max error, cosine sim, SNR, normalized L2 $E_i$, activation-weighted error | Axe 3 |
| `quantization_engine.py` | Real INT8/4/2, Q1_0, MXFP4 quantizers; candidate generator; AWQ calibration | Algo 1, 2, 3 |
| `mixed_precision.py` | Convex-hull greedy & DP multiple-choice knapsack solvers; Pareto sweeps | Algo 1, 4, 10 |
| `expert_profiler.py` | MoE routing traces, frequencies, reuse rates, routing entropy $H$ | Axe 8 |
| `expert_cache.py` | LRU / LFU / FIFO / adaptive cache simulation; $T_{cache}$ formula; size optimizer | Algo 7 |
| `planner.py` | 9-step Z2/D2 decision pipeline + runtime feedback & adaptation loop | Z2/D2 Planner |
| `validation.py` | L1–L3 verification gates against Section 8 success metrics | Axe 11 |

## 5. Documented Validations Reproduced

| Benchmark | Documented | Predicted |
|---|---|---|
| GTX 1080 — Qwen3.5-9B mixed decode | 33.42 tok/s / PPL 9.7839 | 38.52 tok/s / ΔPPL +0.15 |
| GTX 1080 — Q4_0 uniform decode | 35.98 tok/s / PPL 10.0437 | 29.81 tok/s |
| Bonsai-27B Q1_0 footprint | ~3.9 GB weights @ 1.125 bpw | 4.3 GB @ 1.125 bpw |
| HTP Marco-MoE (mobile) | 31.23 tok/s Adreno / 23.6 tok/s HTP | — (profile anchor) |
| Bonsai-27B HTP fallback stress | 99.7% CPU fallback, 0.32 tok/s | parser: 100% fallback, CRITICAL |

## 6. Testing

```bash
PYTHONPATH=src pytest -v tests/
# 20 passed in ~0.4s
```

## 7. Golden Rules (Section 10 of the spec)

1. Never confuse checkpoint size with working set.
2. Never confuse compact format with fast kernel.
3. Never confuse backend compatibility with real acceleration.
4. Never treat perplexity as the only quality metric.
5. Never impose uniform precision.
6. Never quantify all experts the same way without measuring sensitivity.
7. Never assume LRU cache is optimal by default.
8. Always separate requested, transferred, and actually-read bytes.
9. Always measure dequantization cost.
10. Always confront predictions with hardware measurements.
11. Keep GTX 1080/Ti results as experimental references.
12. Keep Qualcomm HTP/Adreno as a first-class hardware axis.
13. Strictly distinguish implemented, validated, and experimental features.

## 8. Status

- Framework version: **1.0.0** (reference implementation)
- Documentation: **18-page technical report** at `doc/main.pdf`
- Validated on: simulated Kimi K3, Qwen3.5-9B, Bonsai-27B, Marco-MoE presets
- Test coverage: 20 automated tests, all passing

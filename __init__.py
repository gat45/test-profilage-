"""Adaptive Quantization & Hardware Profiling Framework.

Implementation of the complete project architecture:
  - Profiler V3/V5 (structural, memory, I/O, MoE profiling, cost prediction)
  - Quantification Algo (mixed precision allocation, sensitivity, Pareto search)
  - Z2/D2 Planner (9-step decision pipeline)
  - Validation L1-L3
"""

__version__ = "1.0.0"

from .capability_db import CapabilityDB, DeviceProfile
from .quant_formats import QuantFormat, FormatRegistry, quant_registry
from .profile_model import ModelMap, TensorMeta
from .profiler import MemoryProfiler, MemoryProfile
from .sensitivity_engine import SensitivityEngine, SensitivityMetrics
from .quantization_engine import QuantizationEngine, QuantCandidate
from .mixed_precision import MixedPrecisionSolver, AllocationPlan
from .expert_profiler import ExpertProfiler, ExpertRoutingStats
from .expert_cache import ExpertCacheSimulator, CacheResult
from .cost_model import CostModel, TokenCostBreakdown
from .predict_from_hf import OfflinePredictor, PredictionReport, predict
from .predict_from_hf_v5 import V5Predictor, V5Report
from .parse_hexagon import HexagonTraceParser, HexagonTrace
from .planner import Z2D2Planner, PlannerResult, ExecutionStrategy
from .validation import ValidationEngine, ValidationReport

__all__ = [
    "CapabilityDB", "DeviceProfile",
    "QuantFormat", "FormatRegistry", "quant_registry",
    "ModelMap", "TensorMeta",
    "MemoryProfiler", "MemoryProfile",
    "SensitivityEngine", "SensitivityMetrics",
    "QuantizationEngine", "QuantCandidate",
    "MixedPrecisionSolver", "AllocationPlan",
    "ExpertProfiler", "ExpertRoutingStats",
    "ExpertCacheSimulator", "CacheResult",
    "CostModel", "TokenCostBreakdown",
    "OfflinePredictor", "PredictionReport", "predict",
    "V5Predictor", "V5Report",
    "HexagonTraceParser", "HexagonTrace",
    "Z2D2Planner", "PlannerResult", "ExecutionStrategy",
    "ValidationEngine", "ValidationReport",
]

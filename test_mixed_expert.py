import pytest
from adaptive_quant.profile_model import ModelMap
from adaptive_quant.mixed_precision import MixedPrecisionSolver
from adaptive_quant.expert_profiler import ExpertProfiler
from adaptive_quant.expert_cache import ExpertCacheSimulator, POLICY_MAP
from adaptive_quant.cost_model import CostModel
from adaptive_quant.capability_db import CapabilityDB
from adaptive_quant.parse_hexagon import HexagonTraceParser
from adaptive_quant.planner import Z2D2Planner
from adaptive_quant.validation import ValidationEngine
from adaptive_quant.profiler import MemoryProfiler


def test_greedy_solver_respects_budget():
    m = ModelMap.qwen35_9b()
    slv = MixedPrecisionSolver()
    budget = int(8.0 * (1024 ** 3))
    plan = slv.solve_greedy(m, budget)
    assert plan.total_size_bytes <= budget
    assert plan.average_bpw > 2.0


def test_mixed_better_than_low_uniform():
    m = ModelMap.qwen35_9b()
    slv = MixedPrecisionSolver()
    mixed = slv.solve_greedy(m, int(8.0 * 1024 ** 3))
    q4 = slv.uniform_plan(m, "Q4_0")
    assert mixed.total_weighted_error < q4.total_weighted_error


def test_pareto_frontier_monotonic():
    m = ModelMap.marco_moe()
    slv = MixedPrecisionSolver()
    plans = slv.solve_pareto(m, n_points=5)
    for a, b in zip(plans, plans[1:]):
        assert a.total_size_bytes <= b.total_size_bytes + 1e6
        assert a.total_weighted_error >= b.total_weighted_error - 1e-6


def test_expert_profiler_entropy():
    ep = ExpertProfiler(8, 2)
    t = ep.synthetic_trace(100, "concentrated", seed=3)
    s = ep.analyze(t)
    assert 0 <= s.entropy_ratio <= 1.0
    assert sum(s.frequencies.values()) == 200


def test_cache_hit_rate_formula():
    sim = ExpertCacheSimulator(bytes_per_expert=100 * 1024 * 1024)
    trace = [[0, 1], [0, 1], [0, 1]]
    r = sim.simulate(trace, 2, "lru")
    assert abs(r.hit_rate - r.hits / (r.hits + r.misses)) < 1e-4


def test_cost_model_breakdown_positive():
    m = ModelMap.qwen35_9b()
    dev = CapabilityDB().get("gtx_1080_8gb")
    b = CostModel().t_token(m, dev)
    for v in (b.t_compute_ms, b.t_dequant_ms, b.t_other_ms, b.t_effective_ms):
        assert v >= 0
    assert b.decode_tok_s > 0


def test_planner_nine_steps(tmp_path):
    m = ModelMap.marco_moe()
    db = CapabilityDB()
    res = Z2D2Planner(db).run(m, "oneplus_15_htp_v81")
    assert len(res.decisions) >= 7
    res.export(str(tmp_path))
    assert (tmp_path / "quantization_plan.json").exists()
    assert (tmp_path / "execution_strategy.json").exists()
    assert (tmp_path / "decision_log.json").exists()


def test_validation_catches_overflow():
    m = ModelMap.qwen35_9b()
    slv = MixedPrecisionSolver()
    plan = slv.uniform_plan(m, "BF16")  # ~16.7 GB plan
    dev = CapabilityDB().get("gtx_1080_8gb")
    mem = MemoryProfiler().profile(m, dev, quant_plan=plan.plan)
    rep = ValidationEngine().validate_plan(m, plan, mem=mem)
    statuses = [c.status for c in rep.checks if c.name == "working_set_within_memory"]
    assert "FAIL" in statuses


def test_hexagon_trace_fallback():
    p = HexagonTraceParser()
    t = p.parse(p.bonsai_stress_trace())
    assert t.cpu_fallback_pct > 90
    assert any("CRITICAL" in n for n in t.notes)


def test_feedback_loop():
    m = ModelMap.qwen35_9b()
    db = CapabilityDB()
    pl = Z2D2Planner(db)
    pl.run(m, "gtx_1080_8gb")
    res = pl.feedback({"measured_decode_tok_s": 33.42, "fallback_pct": 5})
    assert res.measured_feedback is not None

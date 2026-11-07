"""MoE expert profiling (Profiler Axe 8 — MoE-specific profiling).

Ingests/simulates expert routing traces and computes expert frequencies,
reuse rates, routing entropy, and co-activation patterns.
"""

from __future__ import annotations
import math
import random
from collections import Counter
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class ExpertRoutingStats:
    n_experts: int
    top_k: int
    n_tokens: int
    frequencies: Dict[int, int]
    reuse_rates: Dict[int, float]          # Ne / Ntotal
    entropy_nats: float                    # -Σ p log p
    entropy_ratio: float                   # entropy / ln(n_experts) in [0, 1]
    top_expert_share: float                # % selections by most-used expert
    co_activation_sample: Dict[str, int]   # "e1:e2" -> count
    distribution_type: str                 # "concentrated", "balanced", "uniform"

    def to_dict(self) -> dict:
        return asdict(self)


class ExpertProfiler:
    """Analyzes dynamic behavior of MoE router."""

    def __init__(self, n_experts: int, top_k: int) -> None:
        self.n_experts = n_experts
        self.top_k = top_k

    # ------------------------------------------------ trace generation
    def synthetic_trace(
        self,
        n_tokens: int = 200,
        distribution: str = "concentrated",    # "concentrated", "balanced", "uniform"
        seed: int = 42,
    ) -> List[List[int]]:
        """Generate realistic synthetic routing trace."""
        rnd = random.Random(seed)
        if distribution == "uniform":
            weights = [1.0] * self.n_experts
        elif distribution == "concentrated":
            # Zipf-like: top few experts get most traffic
            weights = [1.0 / (i + 1) ** 0.9 for i in range(self.n_experts)]
        else:  # balanced
            weights = [1.0 + rnd.random() * 0.5 for _ in range(self.n_experts)]

        total = sum(weights)
        probs = [w / total for w in weights]
        trace: List[List[int]] = []
        for _ in range(n_tokens):
            picked = set()
            while len(picked) < min(self.top_k, self.n_experts):
                r = rnd.random()
                cum = 0.0
                for e, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        picked.add(e)
                        break
            trace.append(sorted(picked))
        return trace

    # ------------------------------------------------ analysis
    def analyze(self, trace: List[List[int]]) -> ExpertRoutingStats:
        n_tokens = len(trace)
        total_selections = n_tokens * self.top_k
        freq: Counter = Counter()
        co_act: Counter = Counter()
        for tok in trace:
            freq.update(tok)
            for i in range(len(tok)):
                for j in range(i + 1, len(tok)):
                    pair = f"{min(tok[i], tok[j])}:{max(tok[i], tok[j])}"
                    co_act[pair] += 1

        reuse = {e: freq[e] / max(total_selections, 1) for e in range(self.n_experts)}
        # routing entropy
        entropy = 0.0
        for e in range(self.n_experts):
            p = freq[e] / max(total_selections, 1)
            if p > 0:
                entropy -= p * math.log(p)
        max_ent = math.log(max(self.n_experts, 1)) or 1.0
        ratio = entropy / max_ent
        dist_type = "concentrated" if ratio < 0.6 else ("balanced" if ratio < 0.9 else "uniform")

        top_share = (freq.most_common(1)[0][1] / max(total_selections, 1)) if freq else 0.0

        return ExpertRoutingStats(
            n_experts=self.n_experts,
            top_k=self.top_k,
            n_tokens=n_tokens,
            frequencies=dict(freq),
            reuse_rates={e: round(v, 4) for e, v in reuse.items()},
            entropy_nats=round(entropy, 3),
            entropy_ratio=round(ratio, 3),
            top_expert_share=round(top_share, 4),
            co_activation_sample=dict(co_act.most_common(10)),
            distribution_type=dist_type,
        )

    def print_report(self, stats: ExpertRoutingStats) -> str:
        s = stats
        mc = Counter(s.frequencies).most_common(8)
        top_str = ", ".join(f"E{e}:{cnt}" for e, cnt in mc)
        return (
            f"=== MoE Routing Profile ({s.n_tokens} tokens, {s.n_experts} experts, top-{s.top_k}) ===\n"
            f"  distribution : {s.distribution_type} (entropy ratio {s.entropy_ratio:.2f})\n"
            f"  top expert   : E{mc[0][0]} ({s.top_expert_share*100:.1f}% of activations)\n"
            f"  top-8 most-used : {top_str}"
        )

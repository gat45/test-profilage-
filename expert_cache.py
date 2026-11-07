"""MoE expert cache simulation & optimization (Quant Algo 7 + Profiler Axe 8).

Compares LRU / LFU / adaptive / FIFO cache policies against real or synthetic
routing traces, computing hit rates, evictions, loaded bytes, and the cost
T_cache = T_miss + T_load + T_eviction, and finding the optimal cache size.
"""

from __future__ import annotations
import math
from collections import OrderedDict
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple


@dataclass
class CacheResult:
    policy: str
    capacity_experts: int
    cache_size_mb: float
    hits: int
    misses: int
    evictions: int
    hit_rate: float                   # hits / (hits + misses)
    bytes_loaded_mb: float
    mb_per_token: float               # cold IO per token from misses
    cache_time_ms: float              # T_cache model
    n_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


class _BasePolicy:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.cache: set = set()

    def touch(self, expert: int) -> bool:
        """Return True on hit."""
        raise NotImplementedError

    def evict(self) -> int:
        raise NotImplementedError


class LRUPolicy(_BasePolicy):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.order: OrderedDict[int, None] = OrderedDict()

    def touch(self, expert: int) -> bool:
        if expert in self.order:
            self.order.move_to_end(expert)
            return True
        if len(self.order) >= self.capacity:
            self.order.popitem(last=False)
        self.order[expert] = None
        self.cache = set(self.order.keys())
        return False


class LFUPolicy(_BasePolicy):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.freq: Dict[int, int] = {}
        self.entry_order: Dict[int, int] = {}
        self._t = 0

    def touch(self, expert: int) -> bool:
        self._t += 1
        if expert in self.freq:
            self.freq[expert] += 1
            return True
        if len(self.freq) >= self.capacity:
            victim = min(self.freq.keys(), key=lambda e: (self.freq[e], self.entry_order[e]))
            del self.freq[victim]
            del self.entry_order[victim]
        self.freq[expert] = 1
        self.entry_order[expert] = self._t
        self.cache = set(self.freq.keys())
        return False


class FIFOPolicy(_BasePolicy):
    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.q: List[int] = []

    def touch(self, expert: int) -> bool:
        if expert in self.cache:
            return True
        if len(self.q) >= self.capacity:
            victim = self.q.pop(0)
            self.cache.remove(victim)
        self.q.append(expert)
        self.cache.add(expert)
        return False


class AdaptivePolicy(_BasePolicy):
    """ARC-like adaptive policy combining recency and frequency."""

    def __init__(self, capacity: int) -> None:
        super().__init__(capacity)
        self.recent: OrderedDict[int, None] = OrderedDict()   # recency tier
        self.frequent: OrderedDict[int, None] = OrderedDict() # frequency tier
        self.recent_cap = max(1, capacity // 2)

    def touch(self, expert: int) -> bool:
        if expert in self.recent:
            self.recent.pop(expert)
            self.frequent[expert] = None
            self.cache = set(self.recent) | set(self.frequent)
            return True
        if expert in self.frequent:
            self.frequent.move_to_end(expert)
            self.cache = set(self.recent) | set(self.frequent)
            return True
        # Miss: insert in recency
        if len(self.recent) >= self.recent_cap:
            if self.recent:
                self.recent.popitem(last=False)
        elif len(self.recent) + len(self.frequent) >= self.capacity:
            if self.frequent:
                self.frequent.popitem(last=False)
            elif self.recent:
                self.recent.popitem(last=False)
        self.recent[expert] = None
        self.cache = set(self.recent) | set(self.frequent)
        return False


POLICY_MAP = {
    "lru": LRUPolicy,
    "lfu": LFUPolicy,
    "fifo": FIFOPolicy,
    "adaptive": AdaptivePolicy,
}


class ExpertCacheSimulator:
    """Simulates MoE expert caches and optimizes cache sizing."""

    def __init__(
        self,
        bytes_per_expert: float = 300 * 1024 * 1024,      # default ~300 MB per expert (Kimi K3-ish)
        storage_bw_mbps: float = 3000.0,                   # NVMe
        t_miss_ms: float = 0.05,
        t_eviction_ms: float = 0.01,
    ) -> None:
        self.bytes_per_expert = bytes_per_expert
        self.storage_bw_mbps = storage_bw_mbps
        self.t_miss_ms = t_miss_ms
        self.t_eviction_ms = t_eviction_ms

    def simulate(
        self,
        trace: List[List[int]],
        capacity_experts: int,
        policy: str = "lru",
    ) -> CacheResult:
        cache = POLICY_MAP[policy.lower()](capacity_experts)
        hits = misses = evictions = 0
        loaded_bytes = 0.0
        expert_mb = self.bytes_per_expert / (1024 ** 2)

        for tok in trace:
            for expert in tok:
                if expert in cache.cache:
                    cache.touch(expert)
                    hits += 1
                else:
                    pre_size = len(cache.cache)
                    cache.touch(expert)
                    misses += 1
                    loaded_bytes += self.bytes_per_expert
                    if pre_size >= capacity_experts:
                        evictions += 1

        hit_rate = hits / max(hits + misses, 1)
        loaded_mb = loaded_bytes / (1024 ** 2)
        # T_cache = T_miss*misses + T_load(bytes/bw) + T_eviction*evictions
        load_time_ms = (loaded_mb / max(self.storage_bw_mbps, 1.0)) * 1000.0
        cache_time_ms = misses * self.t_miss_ms + load_time_ms + evictions * self.t_eviction_ms

        return CacheResult(
            policy=policy,
            capacity_experts=capacity_experts,
            cache_size_mb=round(capacity_experts * expert_mb, 1),
            hits=hits,
            misses=misses,
            evictions=evictions,
            hit_rate=round(hit_rate, 4),
            bytes_loaded_mb=round(loaded_mb, 1),
            mb_per_token=round(loaded_mb / max(len(trace), 1), 2),
            cache_time_ms=round(cache_time_ms, 1),
            n_tokens=len(trace),
        )

    def compare_policies(
        self,
        trace: List[List[int]],
        capacity_experts: int,
        policies: Optional[List[str]] = None,
    ) -> List[CacheResult]:
        """Run all policies on the same trace."""
        pols = policies or list(POLICY_MAP.keys())
        return [self.simulate(trace, capacity_experts, p) for p in pols]

    def optimize_size(
        self,
        trace: List[List[int]],
        target_hit_rate: float = 0.85,
        max_experts: int = 128,
        policy: str = "adaptive",
    ) -> CacheResult:
        """Find smallest cache reaching target hit rate."""
        for cap in range(1, max_experts + 1):
            r = self.simulate(trace, cap, policy)
            if r.hit_rate >= target_hit_rate:
                return r
        return r

    @staticmethod
    def compare_table(results: List[CacheResult]) -> str:
        hdr = f"{'Policy':<10} {'Cap':>5} {'Size(MB)':>10} {'HitRate':>9} {'Misses':>8} {'Evict':>7} {'MB/token':>10} {'T_cache(ms)':>12}"
        lines = [hdr, "-" * len(hdr)]
        for r in results:
            lines.append(f"{r.policy:<10} {r.capacity_experts:>5} {r.cache_size_mb:>10.1f} "
                         f"{r.hit_rate*100:>8.1f}% {r.misses:>8} {r.evictions:>7} "
                         f"{r.mb_per_token:>10.2f} {r.cache_time_ms:>12.0f}")
        return "\n".join(lines)

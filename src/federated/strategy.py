"""Aggregation strategy (Flower-style Strategy API, in-process simulation).

Supports weighted FedAvg, coordinate-wise median and trimmed mean, optionally
routed through the Secure Aggregation protocol. Robust aggregators need access
to individual updates, so they are rejected when secure aggregation is on.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from ..clients.bank_client import FitResult
from ..secure_aggregation.protocol import secure_sum

AGGREGATORS = ("fedavg", "median", "trimmed_mean")


@dataclass
class AggInfo:
    uplink_bytes: int
    n_aggregated: int
    n_dropped: int


class FedStrategy:
    def __init__(self, aggregator: str = "fedavg", trim_ratio: float = 0.2, server_lr: float = 1.0,
                 secure: bool = False, secagg_clip: float = 16.0):
        if aggregator not in AGGREGATORS:
            raise ValueError(f"aggregator must be one of {AGGREGATORS}")
        if secure and aggregator != "fedavg":
            raise ValueError("Robust aggregators need individual updates and are incompatible "
                             "with secure aggregation; use aggregator='fedavg'.")
        self.aggregator, self.trim_ratio, self.server_lr = aggregator, trim_ratio, server_lr
        self.secure, self.secagg_clip = secure, secagg_clip

    def aggregate_fit(self, round_id: int, results: List[FitResult],
                      dropped: Sequence[int] = ()) -> tuple[np.ndarray, AggInfo]:
        """Return the aggregated *delta* to add to the global weights."""
        dropped = set(dropped)
        alive = [r for r in results if r.cid not in dropped]
        if not alive:
            raise ValueError("All clients dropped")
        P = len(results[0].delta)
        n_all = float(sum(r.n for r in results))
        if self.secure:
            vecs = {r.cid: (r.n / n_all) * r.delta.astype(np.float64) for r in results}
            total = secure_sum(vecs, round_id, tuple(dropped), self.secagg_clip)
            agg = total / sum(r.n / n_all for r in alive)
            info = AggInfo(len(alive) * (P * 8 + 32), len(alive), len(dropped))
        else:
            D = np.stack([r.delta for r in alive]).astype(np.float64)
            w = np.array([r.n for r in alive], dtype=np.float64)
            if self.aggregator == "fedavg":
                agg = (w[:, None] * D).sum(0) / w.sum()
            elif self.aggregator == "median":
                agg = np.median(D, axis=0)
            else:
                k = int(np.floor(self.trim_ratio * len(alive)))
                S = np.sort(D, axis=0)
                agg = S[k:len(alive) - k].mean(0) if len(alive) - 2 * k > 0 else np.median(D, axis=0)
            info = AggInfo(len(alive) * P * 4, len(alive), len(dropped))
        return (self.server_lr * agg).astype(np.float32), info

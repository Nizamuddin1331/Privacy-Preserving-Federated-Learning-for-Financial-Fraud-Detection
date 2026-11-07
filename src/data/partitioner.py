"""Split the training set across simulated banks (IID and Non-IID)."""
from __future__ import annotations

from typing import Dict, List

import numpy as np

STRATEGIES = ("iid", "dirichlet", "prevalence")


def partition(y: np.ndarray, n_clients: int, strategy: str = "iid", alpha: float = 0.5,
              seed: int = 0, min_size: int = 50, min_fraud: int = 3) -> List[np.ndarray]:
    """Return a list of index arrays, one per client.

    iid        : stratified random split, every bank sees the same fraud rate.
    dirichlet  : label-distribution skew. For each class, client shares ~ Dir(alpha).
                 Small alpha -> strongly heterogeneous banks (size and fraud rate).
    prevalence : legitimate rows split evenly, fraud rows split with Dir(alpha), so
                 banks differ mainly in *fraud prevalence* (e.g. a fraud hotspot).
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
    rng = np.random.default_rng(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) < n_clients * min_fraud:
        min_fraud = max(1, len(pos) // (2 * n_clients))

    def split(idx, props):
        idx = rng.permutation(idx)
        cuts = (np.cumsum(props)[:-1] * len(idx)).astype(int)
        return np.split(idx, cuts)

    for _ in range(500):
        if strategy == "iid":
            p_pos = p_neg = np.full(n_clients, 1.0 / n_clients)
        elif strategy == "dirichlet":
            p_pos = rng.dirichlet(np.full(n_clients, alpha))
            p_neg = rng.dirichlet(np.full(n_clients, alpha))
        else:
            p_pos = rng.dirichlet(np.full(n_clients, alpha))
            p_neg = np.full(n_clients, 1.0 / n_clients)
        parts_pos, parts_neg = split(pos, p_pos), split(neg, p_neg)
        parts = [np.concatenate([a, b]) for a, b in zip(parts_pos, parts_neg)]
        if all(len(p) >= min_size for p in parts) and all(len(a) >= min_fraud for a in parts_pos):
            return [rng.permutation(p) for p in parts]
    raise RuntimeError("Could not build a valid partition; increase alpha or data size")


def describe_partition(parts: List[np.ndarray], y: np.ndarray) -> List[Dict]:
    out = []
    for i, p in enumerate(parts):
        nf = int(y[p].sum())
        out.append({"client": i, "n": int(len(p)), "n_fraud": nf,
                    "fraud_rate": float(nf / max(1, len(p)))})
    return out

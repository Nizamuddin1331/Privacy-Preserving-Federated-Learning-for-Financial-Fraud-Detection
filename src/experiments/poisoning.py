"""Model-poisoning / Byzantine robustness: attacks and the evaluation sweep."""
from __future__ import annotations

from dataclasses import replace
from typing import Iterable, List

import numpy as np

ATTACKS = ("none", "label_flip", "sign_flip", "gaussian")


def poison_labels(y: np.ndarray) -> np.ndarray:
    """Data poisoning: attacker trains on flipped labels."""
    return 1.0 - y


def poison_update(delta: np.ndarray, attack: str, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Model poisoning: attacker tampers with the update it sends."""
    if attack == "sign_flip":
        return (-scale * delta).astype(np.float32)
    if attack == "gaussian":
        std = float(np.std(delta)) + 1e-8
        return (rng.normal(0.0, scale * std, size=delta.shape)).astype(np.float32)
    return delta


def robustness_configs(base_cfg, attacks: Iterable[str] = ("sign_flip", "label_flip"),
                       aggregators: Iterable[str] = ("fedavg", "median", "trimmed_mean"),
                       n_malicious: int = 1, include_secagg: bool = True) -> list:
    """Config grid: a clean reference plus attack x aggregator (+ FedAvg under SecAgg)."""
    common = dict(mode="C", dp=False, tag="robust")
    cfgs = [replace(base_cfg, n_malicious=0, attack="none", aggregator="fedavg", secure_aggregation=False,
                    name="robust_clean", **common)]
    for atk in attacks:
        for agg in aggregators:
            cfgs.append(replace(base_cfg, n_malicious=n_malicious, attack=atk, aggregator=agg,
                                secure_aggregation=False, name=f"robust_{atk}_{agg}", **common))
        if include_secagg:
            cfgs.append(replace(base_cfg, n_malicious=n_malicious, attack=atk, aggregator="fedavg",
                                secure_aggregation=True, name=f"robust_{atk}_fedavg_secagg", **common))
    return cfgs


def run_robustness_suite(base_cfg, data=None, progress=None, **grid) -> List[dict]:
    """Run the grid from `robustness_configs`.

    Robust aggregators (median / trimmed mean) need to *see individual updates*,
    which Secure Aggregation hides by design - the grid makes that tension explicit.
    """
    from .runner import run_experiment  # lazy: runner imports this module
    return [run_experiment(c, data=data, progress=progress) for c in robustness_configs(base_cfg, **grid)]

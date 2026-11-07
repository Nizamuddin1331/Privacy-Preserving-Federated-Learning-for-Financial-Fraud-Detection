"""Federated server: owns the global model, runs rounds, evaluates after each one."""
from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

from ..clients.bank_client import BankClient
from ..evaluation.metrics import evaluate_model
from ..models.fraud_mlp import FraudMLP, get_flat, set_flat
from ..privacy.dp_sgd import DPConfig
from .strategy import FedStrategy


class FederatedServer:
    def __init__(self, in_dim: int, hidden: Sequence[int], clients: List[BankClient],
                 strategy: FedStrategy, eval_data: tuple, seed: int = 0,
                 dp: Optional[DPConfig] = None, dropout_prob: float = 0.0, min_survivors: int = 2):
        import torch
        torch.manual_seed(seed)
        self.model = FraudMLP(in_dim, hidden)
        self.clients, self.strategy, self.eval_data, self.dp = clients, strategy, eval_data, dp
        self.dropout_prob, self.min_survivors = dropout_prob, min_survivors
        self.rng = np.random.default_rng(seed + 7)
        self.uplink_total = 0

    def _draw_dropouts(self) -> List[int]:
        ids = [c.cid for c in self.clients]
        if self.dropout_prob <= 0 or len(ids) <= self.min_survivors:
            return []
        drop = [i for i in ids if self.rng.random() < self.dropout_prob]
        while len(ids) - len(drop) < self.min_survivors:
            drop.pop()
        return drop

    def run(self, rounds: int, local_epochs: int, lr: float, batch_size: int,
            progress: Optional[Callable[[int, int], None]] = None) -> List[Dict]:
        history = []
        for r in range(1, rounds + 1):
            t0 = time.time()
            g = get_flat(self.model)
            results = [c.fit(g, local_epochs, lr, batch_size, self.dp) for c in self.clients]
            dropped = self._draw_dropouts()
            delta, info = self.strategy.aggregate_fit(r, results, dropped)
            set_flat(self.model, g + delta)
            self.uplink_total += info.uplink_bytes
            m = evaluate_model(self.model, *self.eval_data)
            m.update(round=r, train_loss=float(np.mean([x.loss for x in results])),
                     n_aggregated=info.n_aggregated, n_dropped=info.n_dropped,
                     epsilon=max(c.epsilon(self.dp.delta) for c in self.clients) if self.dp else None,
                     seconds=time.time() - t0)
            history.append(m)
            if progress:
                progress(r, rounds)
        return history

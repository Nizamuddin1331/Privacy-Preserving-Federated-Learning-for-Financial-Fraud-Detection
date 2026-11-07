"""Simulated financial institution: holds private transactions, trains locally."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch

from ..experiments.poisoning import poison_labels, poison_update
from ..models.fraud_mlp import FraudMLP, get_flat, set_flat, weighted_bce
from ..privacy.accountant import RDPAccountant
from ..privacy.dp_sgd import DPConfig, dp_sgd_step, poisson_batch_indices


@dataclass
class FitResult:
    cid: int
    delta: np.ndarray        # local_weights - global_weights
    n: int
    loss: float
    steps: int
    malicious: bool = False


def train_epochs(model, X: np.ndarray, y: np.ndarray, epochs: int, lr: float, batch_size: int,
                 pos_weight: float, rng: np.random.Generator, momentum: float = 0.9) -> float:
    """Plain (non-private) minibatch SGD. Returns the mean loss of the last epoch."""
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    Xt, yt = torch.as_tensor(X), torch.as_tensor(y)
    model.train()
    last = 0.0
    for _ in range(epochs):
        perm = rng.permutation(len(X))
        losses = []
        for i in range(0, len(X), batch_size):
            idx = torch.as_tensor(perm[i:i + batch_size])
            opt.zero_grad()
            loss = weighted_bce(model(Xt[idx]), yt[idx], pos_weight)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach()))
        last = float(np.mean(losses))
    return last


class BankClient:
    def __init__(self, cid: int, X: np.ndarray, y: np.ndarray, hidden: Sequence[int],
                 pos_weight: float, seed: int = 0, attack: str = "none", attack_scale: float = 5.0):
        self.cid = cid
        self.attack = attack
        self.attack_scale = attack_scale
        self.X = X
        self.y = poison_labels(y) if attack == "label_flip" else y
        self.n = len(X)
        self.pos_weight = pos_weight
        self.model = FraudMLP(X.shape[1], hidden)
        self.rng = np.random.default_rng(seed * 1000 + cid)
        self.gen = torch.Generator().manual_seed(seed * 1000 + cid)
        self.accountant = RDPAccountant()

    @property
    def malicious(self) -> bool:
        return self.attack != "none"

    def fit(self, global_flat: np.ndarray, local_epochs: int, lr: float, batch_size: int,
            dp: Optional[DPConfig] = None) -> FitResult:
        set_flat(self.model, global_flat)
        steps, loss = 0, 0.0
        if dp is None:
            loss = train_epochs(self.model, self.X, self.y, local_epochs, lr, batch_size,
                                self.pos_weight, self.rng)
            steps = local_epochs * int(np.ceil(self.n / batch_size))
        else:
            q = min(1.0, batch_size / self.n)
            steps = local_epochs * int(np.ceil(self.n / batch_size))
            opt = torch.optim.SGD(self.model.parameters(), lr=lr, momentum=0.0)
            Xt, yt = torch.as_tensor(self.X), torch.as_tensor(self.y)
            loss_fn = lambda lg, t: weighted_bce(lg, t, self.pos_weight)  # noqa: E731
            self.model.train()
            losses = []
            for _ in range(steps):
                idx = torch.as_tensor(poisson_batch_indices(self.n, q, self.rng), dtype=torch.long)
                l = dp_sgd_step(self.model, opt, loss_fn, Xt[idx], yt[idx], dp,
                                expected_batch=q * self.n, generator=self.gen)
                if not np.isnan(l):
                    losses.append(l)
            loss = float(np.mean(losses)) if losses else 0.0
            self.accountant.step(q, dp.noise_multiplier, steps)
        delta = get_flat(self.model) - global_flat
        if self.attack in ("sign_flip", "gaussian"):
            delta = poison_update(delta, self.attack, self.attack_scale, self.rng)
        return FitResult(self.cid, delta.astype(np.float32), self.n, loss, steps, self.malicious)

    def epsilon(self, delta: float) -> float:
        return self.accountant.get_epsilon(delta) if self.accountant.total_steps else 0.0

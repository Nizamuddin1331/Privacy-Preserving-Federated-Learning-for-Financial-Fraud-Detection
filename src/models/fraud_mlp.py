"""PyTorch binary fraud classifier and flat-vector helpers for federated exchange."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FraudMLP(nn.Module):
    """Small MLP returning one logit per transaction.

    No BatchNorm on purpose: BatchNorm mixes samples inside a batch, which breaks
    per-sample gradient clipping (DP-SGD) and is fragile under Non-IID data.
    """

    def __init__(self, in_dim: int, hidden: Sequence[int] = (64, 32)):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU()]
            d = h
        layers.append(nn.Linear(d, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def compute_pos_weight(y: np.ndarray, cap: float = 30.0) -> float:
    """neg/pos ratio, capped so a handful of frauds cannot blow up the gradients."""
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    return float(min(cap, neg / max(pos, 1.0)))


def weighted_bce(logits: torch.Tensor, targets: torch.Tensor, pos_weight: float,
                 reduction: str = "mean") -> torch.Tensor:
    pw = torch.as_tensor(pos_weight, dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pw,
                                              reduction=reduction)


def get_flat(model: nn.Module) -> np.ndarray:
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()]).cpu().numpy().astype(np.float32)


def set_flat(model: nn.Module, flat: np.ndarray) -> None:
    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(torch.as_tensor(flat[offset:offset + n], dtype=p.dtype).view_as(p))
            offset += n
    assert offset == len(flat), "flat vector size does not match model"


@torch.no_grad()
def predict_proba(model: nn.Module, X: np.ndarray, batch: int = 8192) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        out.append(torch.sigmoid(model(torch.as_tensor(X[i:i + batch]))).numpy())
    return np.concatenate(out)

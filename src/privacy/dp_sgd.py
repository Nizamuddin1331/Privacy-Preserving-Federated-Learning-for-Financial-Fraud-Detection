"""DP-SGD (Abadi et al., 2016): per-sample clipping + Gaussian noise.

Implemented directly on torch.func (vmap over per-sample gradients), so it has no
dependency on Opacus. Batches are drawn with Poisson sampling (each record is
included independently with probability q), which is what the RDP accountant in
`accountant.py` assumes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap


@dataclass
class DPConfig:
    noise_multiplier: float = 1.1   # sigma
    max_grad_norm: float = 2.0      # clipping bound C
    delta: float = 1e-5


def per_sample_grads(model: nn.Module, loss_fn: Callable, xb: torch.Tensor,
                     yb: torch.Tensor) -> Dict[str, torch.Tensor]:
    params = {k: v.detach() for k, v in model.named_parameters()}

    def sample_loss(p, x, y):
        logit = functional_call(model, p, (x.unsqueeze(0),))
        return loss_fn(logit, y.unsqueeze(0))

    return vmap(grad(sample_loss), in_dims=(None, 0, 0))(params, xb, yb)


def clip_and_noise(grads: Dict[str, torch.Tensor], cfg: DPConfig, expected_batch: float,
                   generator: torch.Generator | None = None) -> Dict[str, torch.Tensor]:
    """Clip each sample's gradient to norm C, sum, add N(0, sigma^2 C^2), rescale."""
    B = next(iter(grads.values())).shape[0]
    out = {}
    if B > 0:
        sq = sum(g.reshape(B, -1).pow(2).sum(dim=1) for g in grads.values())
        factor = (cfg.max_grad_norm / (sq.sqrt() + 1e-6)).clamp(max=1.0)
    for k, g in grads.items():
        if B > 0:
            summed = (g * factor.view(-1, *([1] * (g.dim() - 1)))).sum(dim=0)
        else:
            summed = torch.zeros(g.shape[1:])
        noise = torch.randn(summed.shape, generator=generator) * (cfg.noise_multiplier * cfg.max_grad_norm)
        out[k] = (summed + noise) / expected_batch
    return out


def dp_sgd_step(model: nn.Module, optimizer: torch.optim.Optimizer, loss_fn: Callable,
                xb: torch.Tensor, yb: torch.Tensor, cfg: DPConfig, expected_batch: float,
                generator: torch.Generator | None = None) -> float:
    """One DP-SGD update. An empty Poisson batch still receives noise (required for DP)."""
    if len(xb) > 0:
        g = per_sample_grads(model, loss_fn, xb, yb)
    else:
        g = {k: torch.zeros(0, *p.shape) for k, p in model.named_parameters()}
    noisy = clip_and_noise(g, cfg, expected_batch, generator)
    for name, p in model.named_parameters():
        p.grad = noisy[name]
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    if len(xb) == 0:
        return float("nan")
    with torch.no_grad():
        return float(loss_fn(model(xb), yb))


def poisson_batch_indices(n: int, q: float, rng: np.random.Generator) -> np.ndarray:
    return np.where(rng.random(n) < q)[0]

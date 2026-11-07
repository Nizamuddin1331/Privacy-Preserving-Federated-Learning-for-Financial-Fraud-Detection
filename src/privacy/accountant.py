"""Renyi-DP (moments) accountant for the Sampled Gaussian Mechanism.

Implements the integer-order closed form of Mironov, Talwar & Zhang (2019):

    A_a = sum_{k=0..a} C(a,k) (1-q)^(a-k) q^k exp((k^2 - k) / (2 sigma^2))
    RDP(a) = log(A_a) / (a - 1)

RDP composes additively over steps. The (eps, delta) conversion uses the tighter
bound of Balle et al. (2020):
    eps = RDP(a) + log((a-1)/a) - (log(delta) + log(a)) / (a - 1)
"""
from __future__ import annotations

import math
from typing import List, Sequence, Tuple

import numpy as np

DEFAULT_ORDERS: Tuple[int, ...] = tuple(list(range(2, 65)) + [80, 96, 128, 192, 256])


def _logsumexp(xs: Sequence[float]) -> float:
    m = max(xs)
    if m == -math.inf:
        return -math.inf
    return m + math.log(sum(math.exp(x - m) for x in xs))


def _log_binom(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def rdp_sampled_gaussian(q: float, sigma: float, alpha: int) -> float:
    """RDP of one step of the Sampled Gaussian Mechanism at integer order alpha."""
    if q <= 0:
        return 0.0
    if sigma <= 0:
        return math.inf
    if q >= 1.0:
        return alpha / (2.0 * sigma ** 2)
    terms = []
    for k in range(alpha + 1):
        terms.append(_log_binom(alpha, k) + (alpha - k) * math.log1p(-q)
                     + (k * math.log(q) if k > 0 else 0.0)
                     + (k * k - k) / (2.0 * sigma ** 2))
    return _logsumexp(terms) / (alpha - 1)


def rdp_to_eps(rdp: np.ndarray, orders: Sequence[int], delta: float) -> Tuple[float, int]:
    best, best_a = math.inf, orders[0]
    for r, a in zip(rdp, orders):
        if not math.isfinite(r):
            continue
        eps = r + math.log((a - 1) / a) - (math.log(delta) + math.log(a)) / (a - 1)
        if eps < best:
            best, best_a = eps, a
    return max(best, 0.0), best_a


class RDPAccountant:
    """Tracks cumulative privacy loss. Call `step` once per (batch of) DP-SGD step(s)."""

    def __init__(self, orders: Sequence[int] = DEFAULT_ORDERS):
        self.orders = list(orders)
        self.rdp = np.zeros(len(self.orders))
        self.history: List[Tuple[float, float, int]] = []
        self._cache = {}

    def step(self, q: float, sigma: float, n_steps: int = 1) -> None:
        key = (round(q, 12), round(sigma, 12))
        if key not in self._cache:
            self._cache[key] = np.array([rdp_sampled_gaussian(q, sigma, a) for a in self.orders])
        self.rdp = self.rdp + n_steps * self._cache[key]
        self.history.append((q, sigma, n_steps))

    @property
    def total_steps(self) -> int:
        return sum(s for _, _, s in self.history)

    def get_epsilon(self, delta: float) -> float:
        return rdp_to_eps(self.rdp, self.orders, delta)[0]

    def get_epsilon_and_order(self, delta: float) -> Tuple[float, int]:
        return rdp_to_eps(self.rdp, self.orders, delta)

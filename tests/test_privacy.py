import math

import numpy as np
import torch

from src.models.fraud_mlp import FraudMLP, weighted_bce
from src.privacy.accountant import RDPAccountant, rdp_sampled_gaussian
from src.privacy.dp_sgd import DPConfig, clip_and_noise, per_sample_grads


def test_rdp_no_subsampling_matches_gaussian_mechanism():
    # q = 1 -> plain Gaussian mechanism: RDP(a) = a / (2 sigma^2)
    assert math.isclose(rdp_sampled_gaussian(1.0, 2.0, 8), 8 / 8.0)


def test_rdp_subsampling_amplifies_privacy():
    assert rdp_sampled_gaussian(0.01, 1.0, 8) < rdp_sampled_gaussian(1.0, 1.0, 8) / 50


def test_epsilon_matches_published_reference():
    # TF-Privacy MNIST tutorial: sigma=1.1, batch 256/60000, 60 epochs, delta=1e-5 -> eps ~ 3.0
    # (their classic conversion). Our tighter conversion must be at or below it, and close.
    a = RDPAccountant(); a.step(256 / 60000, 1.1, int(60 * 60000 / 256))
    assert 2.0 < a.get_epsilon(1e-5) < 3.1


def test_epsilon_monotone_in_steps_and_noise():
    def eps(sigma, steps):
        a = RDPAccountant(); a.step(0.05, sigma, steps); return a.get_epsilon(1e-5)
    assert eps(1.0, 100) < eps(1.0, 400)
    assert eps(2.0, 200) < eps(1.0, 200)


def test_epsilon_composes_across_calls():
    a, b = RDPAccountant(), RDPAccountant()
    a.step(0.05, 1.2, 100); a.step(0.05, 1.2, 100); b.step(0.05, 1.2, 200)
    assert math.isclose(a.get_epsilon(1e-5), b.get_epsilon(1e-5), rel_tol=1e-9)


def test_per_sample_grads_match_autograd():
    torch.manual_seed(0)
    m = FraudMLP(6); x = torch.randn(16, 6); y = (torch.rand(16) < 0.4).float()
    lf = lambda lg, t: weighted_bce(lg, t, 3.0)  # noqa: E731
    g = per_sample_grads(m, lf, x, y)
    m.zero_grad(); lf(m(x), y).backward()
    for k, p in m.named_parameters():
        assert torch.allclose(g[k].mean(0), p.grad, atol=1e-5)


def test_clipping_bounds_every_sample_and_noise_scale():
    torch.manual_seed(1)
    m = FraudMLP(6); x = torch.randn(32, 6) * 5; y = torch.ones(32)
    g = per_sample_grads(m, lambda l, t: weighted_bce(l, t, 30.0), x, y)
    cfg = DPConfig(noise_multiplier=1e-9, max_grad_norm=0.5)
    out = clip_and_noise(g, cfg, expected_batch=1.0)          # = sum of clipped grads
    total = torch.cat([v.flatten() for v in out.values()]).norm()
    assert total <= 32 * 0.5 + 1e-4                              # triangle inequality bound
    noisy = clip_and_noise({k: torch.zeros(0, *v.shape[1:]) for k, v in g.items()},
                           DPConfig(2.0, 1.0), expected_batch=1.0)
    flat = torch.cat([v.flatten() for v in noisy.values()])
    assert 1.5 < flat.std() < 2.5                               # ~ sigma * C

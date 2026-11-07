import numpy as np
import pytest

from src.clients.bank_client import BankClient, FitResult
from src.federated.strategy import FedStrategy
from src.models.fraud_mlp import FraudMLP, get_flat
from src.privacy.dp_sgd import DPConfig


def _res(cid, n, delta):
    return FitResult(cid, np.asarray(delta, dtype=np.float32), n, 0.0, 1)


def test_fedavg_is_sample_weighted():
    agg, _ = FedStrategy().aggregate_fit(1, [_res(0, 100, [1, 1]), _res(1, 300, [5, 9])])
    assert np.allclose(agg, [4.0, 7.0])


def test_median_and_trimmed_mean_resist_outlier():
    rs = [_res(i, 10, [1.0]) for i in range(4)] + [_res(4, 10, [1000.0])]
    assert FedStrategy().aggregate_fit(1, rs)[0][0] > 100
    assert abs(FedStrategy("median").aggregate_fit(1, rs)[0][0] - 1.0) < 1e-6
    assert abs(FedStrategy("trimmed_mean", 0.2).aggregate_fit(1, rs)[0][0] - 1.0) < 1e-6


def test_secure_aggregation_matches_plain_fedavg_with_dropout():
    rng = np.random.default_rng(0)
    rs = [_res(i, 50 + 10 * i, rng.normal(size=200)) for i in range(5)]
    plain, _ = FedStrategy().aggregate_fit(1, rs, dropped=[3])
    secure, info = FedStrategy(secure=True).aggregate_fit(1, rs, dropped=[3])
    assert np.allclose(plain, secure, atol=1e-4) and info.n_dropped == 1


def test_robust_aggregators_incompatible_with_secagg():
    with pytest.raises(ValueError):
        FedStrategy("median", secure=True)


def test_dp_client_tracks_privacy_budget(prepared):
    c = BankClient(0, prepared.X_train[:800], prepared.y_train[:800], (16,), 5.0, seed=1)
    g = get_flat(FraudMLP(prepared.n_features, (16,)))
    assert c.epsilon(1e-5) == 0.0
    r1 = c.fit(g, 1, 0.1, 64, DPConfig(1.1, 1.0, 1e-5)); e1 = c.epsilon(1e-5)
    c.fit(g, 1, 0.1, 64, DPConfig(1.1, 1.0, 1e-5)); e2 = c.epsilon(1e-5)
    assert 0 < e1 < e2 and r1.delta.shape == g.shape


def test_sign_flip_attacker_negates_update(prepared):
    X, y = prepared.X_train[:500], prepared.y_train[:500]
    g = get_flat(FraudMLP(prepared.n_features, (16,)))
    honest = BankClient(0, X, y, (16,), 5.0, seed=2).fit(g, 1, 0.1, 64)
    evil = BankClient(0, X, y, (16,), 5.0, seed=2, attack="sign_flip", attack_scale=2.0).fit(g, 1, 0.1, 64)
    assert np.allclose(evil.delta, -2.0 * honest.delta, atol=1e-5) and evil.malicious

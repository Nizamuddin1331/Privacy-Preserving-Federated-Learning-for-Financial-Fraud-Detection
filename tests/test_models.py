import numpy as np
import torch

from src.models.fraud_mlp import (FraudMLP, compute_pos_weight, get_flat, predict_proba, set_flat,
                                  weighted_bce)
from src.evaluation.metrics import best_f1_threshold, compute_metrics


def test_flat_roundtrip():
    a, b = FraudMLP(10), FraudMLP(10)
    set_flat(b, get_flat(a))
    x = torch.randn(5, 10)
    assert torch.allclose(a(x), b(x))


def test_forward_shape_and_probabilities():
    m = FraudMLP(7)
    p = predict_proba(m, np.random.randn(20, 7).astype(np.float32))
    assert p.shape == (20,) and ((p >= 0) & (p <= 1)).all()


def test_pos_weight_is_capped_and_weights_fraud_more():
    y = np.array([0] * 990 + [1] * 10, dtype=np.float32)
    assert compute_pos_weight(y) == 30.0
    lg = torch.zeros(2)
    t = torch.tensor([1.0, 0.0])
    per = weighted_bce(lg, t, 5.0, reduction="none")
    assert per[0] > per[1]


def test_metrics_perfect_and_threshold():
    y = np.array([0, 0, 0, 1, 1]); p = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    m = compute_metrics(y, p, 0.5)
    assert m["precision"] == m["recall"] == m["f1"] == 1.0 and m["pr_auc"] == 1.0
    assert 0.3 < best_f1_threshold(y, p) <= 0.8


def test_diverged_model_is_flagged_not_crashing(prepared):
    from src.evaluation.metrics import evaluate_model
    m = FraudMLP(prepared.n_features)
    with torch.no_grad():
        for p in m.parameters():
            p.fill_(float("nan"))
    out = evaluate_model(m, prepared.X_val, prepared.y_val, prepared.X_test, prepared.y_test)
    assert out["diverged"] is True and out["pr_auc"] < 0.2

"""Fraud-detection metrics. Accuracy is deliberately not the headline number:
with ~0.2% fraud a model that predicts "legitimate" always scores 99.8%."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import (average_precision_score, confusion_matrix,
                             precision_recall_curve, roc_auc_score)

from ..models.fraud_mlp import predict_proba


def best_f1_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximising F1 on a (validation) set."""
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))]) if len(thr) else 0.5


def compute_metrics(y: np.ndarray, p: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    yhat = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, yhat, labels=[0, 1]).ravel()
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    single_class = len(np.unique(y)) < 2
    return {
        "precision": float(precision), "recall": float(recall), "f1": float(f1),
        "pr_auc": float(average_precision_score(y, p)) if not single_class else float("nan"),
        "roc_auc": float(roc_auc_score(y, p)) if not single_class else float("nan"),
        "fpr": float(fp / max(fp + tn, 1)), "fnr": float(fn / max(fn + tp, 1)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "threshold": float(threshold),
    }


def evaluate_model(model, X_val, y_val, X_test, y_test) -> Dict[str, float]:
    """Operating threshold is chosen on validation data, metrics are reported on test.

    A poisoned model can diverge to NaN/inf weights. Its scores are then replaced by an
    uninformative 0.5 (chance-level metrics) and the run is flagged `diverged` instead of crashing.
    """
    p_val, p_test = predict_proba(model, X_val), predict_proba(model, X_test)
    diverged = not (np.isfinite(p_val).all() and np.isfinite(p_test).all())
    p_val, p_test = np.nan_to_num(p_val, nan=0.5), np.nan_to_num(p_test, nan=0.5)
    m = compute_metrics(y_test, p_test, best_f1_threshold(y_val, p_val))
    m["diverged"] = diverged
    return m

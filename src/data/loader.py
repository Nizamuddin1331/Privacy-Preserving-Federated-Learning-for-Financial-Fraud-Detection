"""Dataset loading, validation and synthetic data generation.

The real benchmark is the Kaggle / ULB "Credit Card Fraud Detection" dataset
(284,807 transactions, 492 frauds, columns Time, V1..V28, Amount, Class).
When the CSV is not available, `generate_synthetic` produces a dataset with the
same schema so the whole pipeline (and the test-suite) runs offline.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd

PCA_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = ["Time"] + PCA_COLS + ["Amount"]
TARGET_COL = "Class"
REQUIRED_COLS = FEATURE_COLS + [TARGET_COL]


class DatasetValidationError(ValueError):
    pass


def validate_creditcard_df(df: pd.DataFrame) -> pd.DataFrame:
    """Check schema, labels and missing values. Returns a cleaned copy."""
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise DatasetValidationError(f"Missing required columns: {missing}")
    df = df[REQUIRED_COLS].copy()
    before = len(df)
    df = df.dropna()
    if len(df) == 0:
        raise DatasetValidationError("Dataset is empty after dropping missing values")
    labels = set(np.unique(df[TARGET_COL]).tolist())
    if not labels <= {0, 1}:
        raise DatasetValidationError(f"Class must be binary 0/1, found {sorted(labels)}")
    if len(labels) < 2:
        raise DatasetValidationError("Dataset must contain both fraud and legitimate rows")
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    df.attrs["dropped_rows"] = before - len(df)
    return df.reset_index(drop=True)


def load_creditcard_csv(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return validate_creditcard_df(pd.read_csv(path))


def generate_synthetic(n_samples: int = 40_000, fraud_rate: float = 0.01,
                       seed: int = 42, hard_fraction: float = 0.3) -> pd.DataFrame:
    """Synthetic data with the credit-card schema.

    Legitimate rows: zero-mean Gaussians whose scale decays with component index
    (mimics PCA variance ordering). Fraud rows: mean-shifted on the components
    that are most discriminative in the real data (V3, V4, V10, V11, V12, V14,
    V16, V17 ...). `hard_fraction` of frauds get only half the shift so the task
    is not trivially separable.
    """
    rng = np.random.default_rng(seed)
    n_fraud = max(2, int(round(n_samples * fraud_rate)))
    n_legit = n_samples - n_fraud
    scales = np.linspace(1.9, 0.4, 28)
    legit = rng.normal(0.0, scales, size=(n_legit, 28))
    shift = np.zeros(28)
    for comp, s in {3: -2.5, 4: 2.2, 7: -1.2, 10: -2.8, 11: 2.4, 12: -2.6,
                    14: -3.2, 16: -2.2, 17: -2.6, 18: -1.4}.items():
        shift[comp - 1] = s
    fraud = rng.normal(0.0, scales * 1.15, size=(n_fraud, 28)) + shift
    hard = rng.random(n_fraud) < hard_fraction
    fraud[hard] = rng.normal(0.0, scales * 1.15, size=(int(hard.sum()), 28)) + 0.5 * shift

    amount_legit = np.round(rng.lognormal(3.4, 1.3, n_legit), 2)
    amount_fraud = np.round(rng.lognormal(3.0, 1.8, n_fraud), 2)
    time_legit = rng.uniform(0, 172_800, n_legit)
    time_fraud = rng.uniform(0, 172_800, n_fraud)

    X = np.vstack([legit, fraud])
    df = pd.DataFrame(X, columns=PCA_COLS)
    df.insert(0, "Time", np.concatenate([time_legit, time_fraud]))
    df["Amount"] = np.concatenate([amount_legit, amount_fraud])
    df[TARGET_COL] = np.concatenate([np.zeros(n_legit, int), np.ones(n_fraud, int)])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def load_dataset(path: Optional[str | Path] = None, n_synthetic: int = 40_000,
                 fraud_rate: float = 0.01, seed: int = 42) -> Tuple[pd.DataFrame, str]:
    """Load creditcard.csv if present, otherwise fall back to synthetic data."""
    if path is not None and Path(path).exists():
        return load_creditcard_csv(path), f"csv:{Path(path).name}"
    return generate_synthetic(n_synthetic, fraud_rate, seed), "synthetic"


def dataset_summary(df: pd.DataFrame, source: str) -> dict:
    n_fraud = int(df[TARGET_COL].sum())
    return {
        "source": source,
        "n_rows": int(len(df)),
        "n_features": len(FEATURE_COLS),
        "n_fraud": n_fraud,
        "n_legit": int(len(df) - n_fraud),
        "fraud_rate": float(n_fraud / len(df)),
        "amount_mean": float(df["Amount"].mean()),
        "amount_median": float(df["Amount"].median()),
        "amount_max": float(df["Amount"].max()),
    }

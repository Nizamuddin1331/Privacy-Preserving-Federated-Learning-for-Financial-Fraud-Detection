"""Pre-processing with zero leakage: every statistic is fitted on train only."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .loader import PCA_COLS, TARGET_COL


@dataclass
class Prepared:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_names: List[str]

    @property
    def n_features(self) -> int:
        return self.X_train.shape[1]


def _features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature engineering: log-amount and cyclic hour-of-day; raw Time dropped."""
    out = df[PCA_COLS].copy()
    out["log_amount"] = np.log1p(df["Amount"].clip(lower=0))
    hours = (df["Time"] % 86_400) / 3_600.0
    out["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    out["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    return out


def prepare(df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1,
            seed: int = 42) -> Prepared:
    """Stratified train/val/test split; StandardScaler is fitted on train only."""
    feats = _features(df)
    y = df[TARGET_COL].to_numpy().astype(np.float32)
    X = feats.to_numpy().astype(np.float32)
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed)
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=seed)
    scaler = StandardScaler().fit(X_train)  # <- fit on train only
    tf = lambda a: scaler.transform(a).astype(np.float32)  # noqa: E731
    return Prepared(tf(X_train), y_train, tf(X_val), y_val, tf(X_test), y_test,
                    scaler, list(feats.columns))

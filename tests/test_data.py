import numpy as np
import pandas as pd
import pytest

from src.data.loader import (DatasetValidationError, generate_synthetic, load_creditcard_csv,
                             validate_creditcard_df)
from src.data.partitioner import describe_partition, partition
from src.data.preprocessor import prepare


def test_synthetic_schema_and_rate(small_df):
    assert {"Time", "Amount", "Class", "V1", "V28"} <= set(small_df.columns)
    assert abs(small_df["Class"].mean() - 0.02) < 0.005


def test_validation_rejects_bad_data(small_df):
    with pytest.raises(DatasetValidationError):
        validate_creditcard_df(small_df.drop(columns=["V3"]))
    bad = small_df.copy(); bad.loc[0, "Class"] = 2
    with pytest.raises(DatasetValidationError):
        validate_creditcard_df(bad)
    with pytest.raises(DatasetValidationError):
        validate_creditcard_df(small_df.assign(Class=0))


def test_csv_roundtrip(tmp_path, small_df):
    p = tmp_path / "cc.csv"; small_df.to_csv(p, index=False)
    assert len(load_creditcard_csv(p)) == len(small_df)
    with pytest.raises(FileNotFoundError):
        load_creditcard_csv(tmp_path / "missing.csv")


def test_no_leakage_scaler_fit_on_train_only(small_df, prepared):
    # train features are standardised exactly; test features are not forced to mean 0
    assert np.allclose(prepared.X_train.mean(0), 0, atol=1e-4)
    assert np.allclose(prepared.X_train.std(0), 1, atol=1e-3)
    assert not np.allclose(prepared.X_test.mean(0), 0, atol=1e-6)
    assert prepared.scaler.n_samples_seen_ == len(prepared.X_train)


def test_split_is_stratified_and_disjoint(small_df, prepared):
    rates = [prepared.y_train.mean(), prepared.y_val.mean(), prepared.y_test.mean()]
    assert max(rates) - min(rates) < 0.01
    assert len(prepared.X_train) + len(prepared.X_val) + len(prepared.X_test) == len(small_df)


@pytest.mark.parametrize("strategy", ["iid", "dirichlet", "prevalence"])
def test_partition_covers_all_rows_once(prepared, strategy):
    parts = partition(prepared.y_train, 4, strategy, alpha=1.0, seed=3)
    idx = np.concatenate(parts)
    assert len(idx) == len(prepared.y_train) and len(set(idx.tolist())) == len(idx)


def test_noniid_is_more_heterogeneous_than_iid(prepared):
    y = prepared.y_train
    spread = lambda s: np.ptp([d["fraud_rate"] for d in describe_partition(partition(y, 5, s, 0.3, 0), y)])  # noqa: E731
    assert spread("prevalence") > spread("iid")


def test_partition_rejects_unknown_strategy(prepared):
    with pytest.raises(ValueError):
        partition(prepared.y_train, 3, "bogus")

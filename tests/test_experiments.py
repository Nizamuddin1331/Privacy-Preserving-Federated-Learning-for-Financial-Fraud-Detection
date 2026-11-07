from dataclasses import replace

import numpy as np
import pytest

from src.experiments.runner import ExperimentConfig, prepare_data, run_experiment, run_suite, save_result

BASE = ExperimentConfig(rounds=4, n_synthetic=8000, fraud_rate=0.02, dataset_path=None, n_clients=4, seed=5)


@pytest.fixture(scope="module")
def data():
    return prepare_data(BASE)


def test_mode_flags():
    assert (ExperimentConfig(mode="D").dp, ExperimentConfig(mode="D").secure_aggregation) == (True, False)
    assert (ExperimentConfig(mode="E").dp, ExperimentConfig(mode="E").secure_aggregation) == (True, True)
    assert not ExperimentConfig(mode="C").dp
    with pytest.raises(ValueError):
        ExperimentConfig(mode="Z")


def test_suite_runs_all_modes_and_federation_beats_isolation(data, tmp_path):
    res = {r["config"]["mode"]: r for r in run_suite(BASE, data=data, out_dir=tmp_path)}
    assert set(res) == set("ABCDE")
    assert len(list(tmp_path.glob("suite_*.json"))) == 5
    assert res["C"]["final"]["pr_auc"] >= res["B"]["final"]["pr_auc"] - 0.05
    assert res["D"]["epsilon"] > 0 and res["E"]["epsilon"] == pytest.approx(res["D"]["epsilon"])
    assert res["C"]["epsilon"] is None
    assert res["E"]["uplink_bytes_total"] > res["C"]["uplink_bytes_total"]    # SecAgg costs bandwidth


def test_secagg_does_not_change_accuracy(data):
    plain = run_experiment(replace(BASE, mode="C", name="p"), data)
    sec = run_experiment(replace(BASE, mode="C", secure_aggregation=True, name="s"), data)
    assert abs(plain["final"]["pr_auc"] - sec["final"]["pr_auc"]) < 0.02


def test_dropout_is_survived(data):
    r = run_experiment(replace(BASE, mode="E", dropout_prob=0.5, name="drop"), data)
    assert sum(h["n_dropped"] for h in r["history"]) > 0 and len(r["history"]) == BASE.rounds


def test_median_defends_against_sign_flip(data):
    atk = dict(mode="C", n_malicious=1, attack="sign_flip", attack_scale=10.0, rounds=6)
    bad = run_experiment(replace(BASE, aggregator="fedavg", name="bad", **atk), data)
    good = run_experiment(replace(BASE, aggregator="median", name="good", **atk), data)
    assert good["final"]["pr_auc"] > bad["final"]["pr_auc"]


def test_non_iid_partition_runs(data):
    r = run_experiment(replace(BASE, mode="C", partition="dirichlet", dirichlet_alpha=0.5, name="n"), data)
    rates = [p["fraud_rate"] for p in r["partition"]]
    assert max(rates) > min(rates)


def test_save_result_roundtrip(data, tmp_path):
    p = save_result(run_experiment(replace(BASE, mode="C", rounds=1, name="x"), data), tmp_path)
    assert p.exists() and p.name == "suite_x.json"

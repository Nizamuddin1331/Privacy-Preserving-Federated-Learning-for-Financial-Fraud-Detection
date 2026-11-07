"""One command to reproduce every result and figure:  python -m src.experiments.run_all"""
from __future__ import annotations

import argparse
from dataclasses import replace

from ..evaluation.visualizer import generate_all
from .poisoning import run_robustness_suite
from .runner import ExperimentConfig, prepare_data, run_noise_sweep, run_suite, save_result


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--partition", default="iid", choices=["iid", "dirichlet", "prevalence"])
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--out", default="results")
    ap.add_argument("--reports", default="reports")
    ap.add_argument("--skip", nargs="*", default=[], choices=["suite", "sweep", "robust"])
    a = ap.parse_args(argv)
    base = ExperimentConfig(rounds=a.rounds, n_clients=a.clients, partition=a.partition, dataset_path=a.data)
    data = prepare_data(base)
    print("dataset:", data.summary)
    if "suite" not in a.skip:
        for r in run_suite(base, data=data, out_dir=a.out):
            print(f"[suite] {r['mode_label']:<24} PR-AUC={r['final']['pr_auc']:.3f} eps={r['epsilon']}")
    if "sweep" not in a.skip:
        for r in run_noise_sweep(base, data=data, out_dir=a.out):
            print(f"[sweep] sigma={r['config']['noise_multiplier']:<4} eps={r['epsilon']:.2f} PR-AUC={r['final']['pr_auc']:.3f}")
    if "robust" not in a.skip:
        rb = replace(base, partition="iid")
        for r in run_robustness_suite(rb, data=data):
            save_result(r, a.out)
            c = r["config"]
            print(f"[robust] {c['name']:<34} PR-AUC={r['final']['pr_auc']:.3f}")
    for p in generate_all(a.out, a.reports):
        print("wrote", p)


if __name__ == "__main__":
    main()

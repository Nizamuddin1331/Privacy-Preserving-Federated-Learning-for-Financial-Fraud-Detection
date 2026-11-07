"""Multi-mode experiment runner.

Modes
  A  centralised baseline (all data pooled - the upper bound, no privacy)
  B  local-only (each bank trains alone - the lower bound)
  C  federated averaging (raw updates visible to the server)
  D  federated + DP-SGD  (record-level (eps, delta)-DP at every bank)
  E  federated + DP-SGD + Secure Aggregation (server only sees the masked sum)

`dp` / `secure_aggregation` can be overridden to build ablations, e.g. mode C with
secure_aggregation=True to verify that SecAgg does not change accuracy.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch

from ..clients.bank_client import BankClient, train_epochs
from ..data.loader import dataset_summary, load_dataset
from ..data.partitioner import describe_partition, partition
from ..data.preprocessor import Prepared, prepare
from ..evaluation.metrics import evaluate_model
from ..federated.server import FederatedServer
from ..federated.strategy import FedStrategy
from ..models.fraud_mlp import FraudMLP, compute_pos_weight
from ..privacy.dp_sgd import DPConfig

MODES = ("A", "B", "C", "D", "E")
MODE_LABELS = {"A": "Centralised", "B": "Local-only", "C": "FedAvg",
               "D": "FedAvg + DP", "E": "FedAvg + DP + SecAgg"}


@dataclass
class ExperimentConfig:
    mode: str = "C"
    name: str = ""
    tag: str = "suite"
    # data
    dataset_path: Optional[str] = "data/creditcard.csv"
    n_synthetic: int = 40_000
    fraud_rate: float = 0.01
    seed: int = 42
    # federation
    n_clients: int = 5
    partition: str = "iid"           # iid | dirichlet | prevalence
    dirichlet_alpha: float = 0.5
    rounds: int = 15
    local_epochs: int = 2
    batch_size: int = 256
    lr: float = 0.05
    dp_lr: float = 0.5               # DP-SGD needs a larger step: clipped, noisy, no momentum
    hidden: Sequence[int] = (64, 32)
    dropout_prob: float = 0.0
    # privacy
    dp: Optional[bool] = None
    noise_multiplier: float = 1.1
    max_grad_norm: float = 2.0
    delta: float = 1e-5
    secure_aggregation: Optional[bool] = None
    # robustness
    aggregator: str = "fedavg"       # fedavg | median | trimmed_mean
    n_malicious: int = 0
    attack: str = "none"             # none | label_flip | sign_flip | gaussian
    attack_scale: float = 5.0

    def __post_init__(self):
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.dp is None:
            self.dp = self.mode in ("D", "E")
        if self.secure_aggregation is None:
            self.secure_aggregation = self.mode == "E"
        if not self.name:
            self.name = f"{self.mode}"
        self.hidden = tuple(self.hidden)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["hidden"] = list(self.hidden)
        return d


@dataclass
class PreparedData:
    prepared: Prepared
    source: str
    summary: dict


def prepare_data(cfg: ExperimentConfig) -> PreparedData:
    df, source = load_dataset(cfg.dataset_path, cfg.n_synthetic, cfg.fraud_rate, cfg.seed)
    return PreparedData(prepare(df, seed=cfg.seed), source, dataset_summary(df, source))


def _cm(m: dict) -> List[List[int]]:
    return [[m["tn"], m["fp"]], [m["fn"], m["tp"]]]


def run_experiment(cfg: ExperimentConfig, data: Optional[PreparedData] = None,
                   progress: Optional[Callable[[int, int], None]] = None) -> dict:
    t0 = time.time()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    data = data or prepare_data(cfg)
    P = data.prepared
    eval_data = (P.X_val, P.y_val, P.X_test, P.y_test)
    pos_weight = compute_pos_weight(P.y_train)
    parts = partition(P.y_train, cfg.n_clients, cfg.partition, cfg.dirichlet_alpha, cfg.seed)
    part_info = describe_partition(parts, P.y_train)
    dp = DPConfig(cfg.noise_multiplier, cfg.max_grad_norm, cfg.delta) if cfg.dp else None
    n_params = sum(p.numel() for p in FraudMLP(P.n_features, cfg.hidden).parameters())
    rng = np.random.default_rng(cfg.seed)
    history: List[dict] = []
    uplink = 0

    if cfg.mode == "A":
        model = FraudMLP(P.n_features, cfg.hidden)
        for r in range(1, cfg.rounds + 1):
            loss = train_epochs(model, P.X_train, P.y_train, cfg.local_epochs, cfg.lr,
                                cfg.batch_size, pos_weight, rng)
            m = evaluate_model(model, *eval_data)
            m.update(round=r, train_loss=loss, epsilon=None)
            history.append(m)
            progress and progress(r, cfg.rounds)
        final_cm = _cm(history[-1])
    elif cfg.mode == "B":
        torch.manual_seed(cfg.seed)
        models = [FraudMLP(P.n_features, cfg.hidden) for _ in parts]
        per_client_final = []
        for r in range(1, cfg.rounds + 1):
            ms = []
            for i, (idx, model) in enumerate(zip(parts, models)):
                loss = train_epochs(model, P.X_train[idx], P.y_train[idx], cfg.local_epochs, cfg.lr,
                                    cfg.batch_size, pos_weight, rng)
                ms.append(evaluate_model(model, *eval_data))
            keys = [k for k in ms[0] if k not in ("tn", "fp", "fn", "tp", "diverged")]
            avg = {k: float(np.nanmean([m[k] for m in ms])) for k in keys}
            for k in ("tn", "fp", "fn", "tp"):
                avg[k] = int(round(np.mean([m[k] for m in ms])))
            avg.update(round=r, train_loss=loss, epsilon=None, diverged=any(m["diverged"] for m in ms))
            history.append(avg)
            per_client_final = ms
            progress and progress(r, cfg.rounds)
        final_cm = _cm(history[-1])
    else:
        clients = []
        for i, idx in enumerate(parts):
            atk = cfg.attack if i < cfg.n_malicious else "none"
            clients.append(BankClient(i, P.X_train[idx], P.y_train[idx], cfg.hidden, pos_weight,
                                      seed=cfg.seed, attack=atk, attack_scale=cfg.attack_scale))
        strategy = FedStrategy(cfg.aggregator, secure=bool(cfg.secure_aggregation))
        server = FederatedServer(P.n_features, cfg.hidden, clients, strategy, eval_data,
                                 seed=cfg.seed, dp=dp, dropout_prob=cfg.dropout_prob)
        history = server.run(cfg.rounds, cfg.local_epochs, cfg.dp_lr if cfg.dp else cfg.lr,
                             cfg.batch_size, progress)
        uplink = server.uplink_total
        final_cm = _cm(history[-1])

    final = {k: v for k, v in history[-1].items()}
    result = {
        "config": cfg.to_dict(),
        "mode_label": MODE_LABELS[cfg.mode],
        "dataset": data.summary,
        "partition": part_info,
        "n_params": n_params,
        "pos_weight": pos_weight,
        "history": history,
        "final": final,
        "confusion_matrix": final_cm,
        "epsilon": final.get("epsilon"),
        "uplink_bytes_total": uplink,
        "runtime_seconds": time.time() - t0,
    }
    if cfg.mode == "B":
        result["per_client_final"] = per_client_final
    return result


def save_result(result: dict, out_dir: str | Path = "results") -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = result["config"]
    path = out_dir / f"{cfg['tag']}_{cfg['name']}.json"
    path.write_text(json.dumps(result, indent=2, default=float))
    return path


def run_suite(base: ExperimentConfig, modes: Sequence[str] = MODES, data: Optional[PreparedData] = None,
              out_dir: str | Path | None = "results", progress=None) -> List[dict]:
    data = data or prepare_data(base)
    results = []
    for m in modes:
        cfg = replace(base, mode=m, name=f"mode_{m}", tag="suite", dp=None, secure_aggregation=None)
        res = run_experiment(cfg, data, progress)
        results.append(res)
        out_dir and save_result(res, out_dir)
    return results


def run_noise_sweep(base: ExperimentConfig, multipliers: Sequence[float] = (0.5, 0.8, 1.1, 1.5, 2.0, 3.0),
                    data: Optional[PreparedData] = None, out_dir: str | Path | None = "results") -> List[dict]:
    """Privacy-utility trade-off: run mode D for several noise multipliers."""
    data = data or prepare_data(base)
    results = []
    for s in multipliers:
        cfg = replace(base, mode="D", noise_multiplier=s, name=f"sigma_{s}", tag="sweep", dp=True,
                      secure_aggregation=False)
        res = run_experiment(cfg, data)
        results.append(res)
        out_dir and save_result(res, out_dir)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description="Run federated fraud-detection experiments")
    ap.add_argument("--modes", default="ABCDE")
    ap.add_argument("--rounds", type=int, default=15)
    ap.add_argument("--clients", type=int, default=5)
    ap.add_argument("--partition", default="iid", choices=["iid", "dirichlet", "prevalence"])
    ap.add_argument("--noise", type=float, default=1.1)
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--out", default="results")
    a = ap.parse_args(argv)
    base = ExperimentConfig(rounds=a.rounds, n_clients=a.clients, partition=a.partition,
                            noise_multiplier=a.noise, dataset_path=a.data)
    for r in run_suite(base, tuple(a.modes), out_dir=a.out):
        f = r["final"]
        print(f"{r['mode_label']:<24} PR-AUC={f['pr_auc']:.3f} F1={f['f1']:.3f} "
              f"recall={f['recall']:.3f} precision={f['precision']:.3f} eps={r['epsilon']}")


if __name__ == "__main__":
    main()

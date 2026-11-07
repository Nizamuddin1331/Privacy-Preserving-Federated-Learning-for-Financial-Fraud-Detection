# Privacy-Preserving Federated Learning for Financial Fraud Detection

MTech capstone project. Several simulated banks train one fraud-detection model **without sharing
transactions**, and the system measures what each added privacy layer costs in detection quality.

| Mode | Training | What the server can see |
|---|---|---|
| A | Centralised (all data pooled) | every raw transaction |
| B | Local-only (no collaboration) | nothing |
| C | Federated averaging (FedAvg) | each bank's raw model update |
| D | FedAvg + **DP-SGD** | updates trained with clipped, noised gradients, with a tracked (ε, δ) |
| E | FedAvg + DP-SGD + **Secure Aggregation** | only the masked sum of updates |

Plus a **poisoning study**: one dishonest bank attacks the model, and FedAvg is compared with coordinate-wise
median and trimmed-mean aggregation (and with SecAgg, which blocks those defences).

The full write-up is in [`docs/REPORT.md`](docs/REPORT.md); generated figures and tables are in `reports/`.

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

pytest -q                                  # 49 tests
python -m src.experiments.run_all          # modes A-E, privacy sweep, poisoning grid, figures
python -m src.experiments.runner --modes CDE --rounds 10     # or a single custom run
```

Real data: download the Kaggle *Credit Card Fraud Detection* file and save it as `data/creditcard.csv`.
If it is missing, a synthetic dataset with the same schema (Time, V1-V28, Amount, Class) is generated, so everything
runs offline. Results are written to `results/*.json`, figures and `summary.md` to `reports/`.

### Dashboard

```bash
cd backend  && uvicorn app.main:app --port 8000        # API docs at http://localhost:8000/docs
cd frontend && npm install && npm run dev               # http://localhost:5173
```

`DATABASE_URL` defaults to SQLite; set a PostgreSQL URL in `.env` (see `.env.example`) for production.

## Layout

```
backend/app/    FastAPI: api/ (health, dataset, experiments, metrics), core/ (config, logger, jobs),
                db/ (SQLAlchemy session + models), schemas/
frontend/src/   React 18 + TypeScript + Vite + Recharts, 10 pages
src/data/       loader (validation, synthetic generator), preprocessor (zero-leakage), partitioner (IID / Non-IID)
src/models/     FraudMLP (PyTorch) with class-weighted loss
src/clients/    BankClient: local training, DP-SGD, optional attacker behaviour
src/federated/  server (rounds, evaluation) and strategy (FedAvg / median / trimmed mean, SecAgg routing)
src/privacy/    dp_sgd (per-sample clipping + Gaussian noise), accountant (RDP accountant)
src/secure_aggregation/  X25519 pairwise masking over Z_(2^64), dropout recovery
src/experiments/         runner (modes A-E), poisoning (attacks + grid), run_all
src/evaluation/          metrics, visualizer
tests/          data, models, federated, privacy, secure aggregation, experiments, API
```

## Differences from the original structure

* The federated engine is an in-process simulation with a Flower-style `Strategy` API, **not** the `flwr` package.
  Flower is awkward to run in a single process with custom DP/SecAgg hooks; the API shape is kept so a
  Flower adapter is a small addition.
* DP-SGD is implemented directly with `torch.func` (per-sample gradients) and a self-written RDP accountant instead of
  Opacus. The accountant is checked against the published TF-Privacy MNIST figure in `tests/test_privacy.py`.
* Added `backend/app/core/jobs.py` (background runner) and `src/experiments/run_all.py` (one-command reproduction).

## Honest limitations

* Secure aggregation is simplified: no Shamir secret sharing, no double-masking, honest-but-curious server.
  It is a faithful demonstration of the masking idea, not a production protocol.
* DP gives **record-level** privacy per bank (every bank's accountant composes over rounds, and ε is the worst bank).
  Local DP-SGD noise is added by each bank, so there is no extra amplification from SecAgg.
* The class-imbalance weight is computed from global training labels and treated as a public constant.
* The numbers in `reports/summary.md` come from synthetic data and a single seed. They show the pipeline working;
  rerun on `creditcard.csv`, with several seeds, before quoting results.

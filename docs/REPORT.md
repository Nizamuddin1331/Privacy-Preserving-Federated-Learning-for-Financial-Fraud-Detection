# Privacy-Preserving Federated Learning for Financial Fraud Detection

*MTech Capstone Project Report*

## Abstract

Banks hold the data needed to detect fraud but cannot pool it because of privacy law and competitive risk. This project builds and evaluates
a federated learning system in which simulated banks jointly train a neural fraud classifier while keeping transactions local. Three
protections are layered and measured separately: (1) federated averaging, (2) record-level differential privacy through DP-SGD with a
Rényi-DP accountant, and (3) pairwise-masking secure aggregation. A poisoning study shows how one dishonest bank can destroy a FedAvg model, how median
and trimmed-mean aggregation resist it, and why those defences conflict with secure aggregation. On the synthetic benchmark used for the reported runs, the
fully protected configuration (ε ≈ 7.3, δ = 10⁻⁵, masked aggregation) kept a PR-AUC of 0.801 against 0.817 for unprotected FedAvg.

## 1. Problem statement and objectives

Fraud is rare (0.17% of transactions in the public benchmark), so each bank sees few positive examples and a pooled model would be better.
Sharing raw transactions is not allowed, and even sharing model updates can leak information about individual records.

Objectives:

1. Build a reproducible federated pipeline for highly imbalanced tabular fraud data, with IID and Non-IID bank splits.
2. Add formal privacy (DP-SGD, tracked as cumulative (ε, δ)) and cryptographic protection of updates (secure aggregation).
3. Quantify the utility cost of each protection against centralised and local-only baselines.
4. Evaluate robustness to model and data poisoning, and the interaction with secure aggregation.
5. Deliver the work as a tested codebase with an API and a dashboard.

## 2. Background

* **FedAvg** (McMahan et al., 2017): clients train locally and the server averages their updates, weighted by data size.
* **DP-SGD** (Abadi et al., 2016): clip each example's gradient to norm C and add Gaussian noise N(0, σ²C²); the privacy loss is tracked with a
  moments / Rényi-DP accountant (Mironov, 2017; Mironov, Talwar & Zhang, 2019), converted to (ε, δ) with the tighter bound of Balle et al. (2020).
* **Secure aggregation** (Bonawitz et al., 2017): clients add pairwise masks that cancel in the sum, so the server learns only the aggregate.
* **Byzantine-robust aggregation**: coordinate-wise median and trimmed mean (Yin et al., 2018) bound the influence of a few malicious updates.
* **Data**: ULB / Kaggle credit-card fraud dataset (Dal Pozzolo et al., 2015): 284,807 transactions, 492 frauds, 28 anonymised PCA features, Time and Amount.

## 3. System design

```
 Bank 1 ─┐  local data never leaves
 Bank 2 ─┼─► train locally (SGD or DP-SGD) ─► update Δᵢ ─► [mask with pairwise secrets] ─► Server ─► aggregate ─► new global model
 Bank N ─┘                                                                       (FedAvg | median | trimmed mean)
```

**Data.** `loader.py` validates schema, labels and missing values, and can generate a synthetic dataset with the same schema. `preprocessor.py` creates
log-amount and cyclic hour-of-day features and performs a stratified train/validation/test split; the scaler is fitted on the training split only.
`partitioner.py` provides IID, Dirichlet label-skew and fraud-prevalence-skew splits.

**Model.** `FraudMLP`: 31 inputs → 64 → 32 → 1 logit, ReLU, no BatchNorm (it breaks per-example clipping). Loss is binary cross-entropy with a positive-class weight
(neg/pos, capped at 30).

**Differential privacy.** Each step draws a Poisson batch (sampling rate q = batch/n), computes per-example gradients with `torch.func.vmap`, clips them to C, sums, adds N(0, σ²C²) and divides by the expected batch
size. `RDPAccountant` implements the integer-order sampled-Gaussian RDP, composes it additively over steps and converts to (ε, δ). Each bank keeps its own accountant; the reported ε is the maximum over banks.

**Secure aggregation.** Each round every bank creates an X25519 key pair; pairwise seeds come from ECDH + HKDF-SHA256 and are expanded by a PRG. Updates are scaled by nᵢ/Σn, encoded in fixed point
(2²⁰ scale) in the ring ℤ₂⁶⁴ and masked: the lower-id bank of each pair adds the mask, the higher-id bank subtracts it. For banks that drop after masking, survivors reveal their seeds with the dropped banks and the server cancels the leftovers.

**Evaluation.** Precision, recall, F1, PR-AUC, ROC-AUC, FPR, FNR and the confusion matrix on a held-out test set. The decision threshold maximises F1 on a validation split. PR-AUC is the headline metric because accuracy is meaningless at this imbalance.

**Software.** FastAPI back end (SQLAlchemy; PostgreSQL or SQLite) launches experiments as background jobs and serves comparison, convergence, privacy, secure-aggregation and robustness views to a React dashboard.

## 4. Experimental setup

Synthetic data with the credit-card schema (40,000 transactions, 1% fraud, 30% of frauds deliberately hard), 5 banks with an IID split, 15 rounds of 2 local epochs, batch 256, hidden layers (64, 32).
Non-private SGD: lr 0.05, momentum 0.9. DP-SGD: σ = 1.1, C = 2, lr 0.5, δ = 10⁻⁵. Seed 42. The poisoning study uses one malicious bank of five (sign-flip attack scaled ×5, or label flipping).
Reproduce with `python -m src.experiments.run_all --data none`.

## 5. Results

### 5.1 Utility of each mode (test set)

| Mode | Precision | Recall | F1 | PR-AUC | ε | Uplink (MB) |
|---|---|---|---|---|---|---|
| A: Centralised | 1.000 | 0.588 | 0.740 | 0.807 | – | – |
| B: Local-only | 0.963 | 0.645 | 0.772 | 0.803 | – | – |
| C: FedAvg | 0.935 | 0.725 | 0.817 | 0.817 | – | 1.25 |
| D: FedAvg + DP | 0.931 | 0.675 | 0.783 | 0.801 | 7.29 | 1.25 |
| E: FedAvg + DP + SecAgg | 0.931 | 0.675 | 0.783 | 0.801 | 7.29 | 2.50 |

Modes D and E give identical accuracy, as expected: secure aggregation computes the same sum, to within fixed-point rounding. Its cost is bandwidth (64-bit masked values, 2× the uplink here). A, B and C are within about 0.015 PR-AUC of each other, which is
inside what one seed on a small synthetic test set (about 80 frauds) can produce, so **this run does not show a federation advantage over local training or centralisation**; the synthetic task is easy enough for a single bank
with 5,600 records. That comparison needs the real dataset and multiple seeds, where fraud counts per bank are far smaller.

### 5.2 Privacy-utility trade-off (mode D)

| σ | ε (δ = 10⁻⁵) | PR-AUC | F1 | Recall |
|---|---|---|---|---|
| 0.5 | 80.2 | 0.814 | 0.782 | 0.650 |
| 0.8 | 15.3 | 0.808 | 0.779 | 0.662 |
| 1.1 | 7.3 | 0.801 | 0.783 | 0.675 |
| 1.5 | 4.4 | 0.791 | 0.777 | 0.675 |
| 2.0 | 2.9 | 0.774 | 0.765 | 0.650 |
| 3.0 | 1.75 | 0.742 | 0.729 | 0.637 |

Quality degrades gently until ε ≈ 3 and more steeply below it. The ε values are for record-level privacy at the worst bank, over 15 rounds.

### 5.3 Poisoning robustness (one malicious bank of five)

| Attack | FedAvg | Median | Trimmed mean | FedAvg + SecAgg |
|---|---|---|---|---|
| none (reference) | 0.817 | – | – | – |
| sign-flip ×5 | 0.010 | 0.811 | 0.816 | 0.013 |
| label-flip | 0.010 | 0.818 | 0.842 | 0.010 |

(PR-AUC. 0.010 is the base fraud rate, i.e. a destroyed model.) One bank is enough to wreck plain FedAvg. Median and trimmed mean fully recover. Secure aggregation offers no defence and, because the server cannot inspect individual updates,
excludes those defences, so it must be combined with other measures (for example update-norm bounds proven in zero knowledge, or client-side checks), which are outside this project.

## 6. Discussion and limitations

* Privacy layers compose cleanly: DP bounds what the trained model reveals about a record, secure aggregation hides individual updates in transit. They do not defend against poisoning.
* ε values depend on the threat model (record-level, local DP-SGD, worst bank). They are not comparable with user-level or central-DP figures.
* Secure aggregation is a simplified, honest-but-curious implementation; the dropout step reveals pairwise seeds of dropped banks only.
* The reported numbers come from synthetic data and a single seed; the real benchmark and repeated seeds are needed before drawing conclusions about federation gains.
* An early DP configuration (clip norm 1, lr 0.05) failed to learn on small data. Clipping with a large positive-class weight changes the effective objective, and DP-SGD needs a larger learning rate and clip norm. The reported defaults come from a small grid on the same data, so the DP numbers are slightly optimistic.

## 7. Conclusion and future work

The system demonstrates that a bank consortium can train a fraud model with formal (ε, δ) privacy and masked aggregation at a small cost in PR-AUC on this benchmark, and that robustness
needs separate mechanisms. Future work: run on the real dataset with 10+ seeds and Non-IID splits; add Shamir-shared seeds and double masking; user-level DP; adaptive clipping; client sampling with privacy amplification;
a Flower adapter for multi-process deployment; and membership-inference attacks to measure empirical leakage against the theoretical ε.

## References

1. McMahan, B. et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.
2. Abadi, M. et al. (2016). Deep Learning with Differential Privacy. ACM CCS.
3. Mironov, I. (2017). Rényi Differential Privacy. IEEE CSF.
4. Mironov, I., Talwar, K., Zhang, L. (2019). Rényi Differential Privacy of the Sampled Gaussian Mechanism. arXiv:1908.10530.
5. Balle, B. et al. (2020). Hypothesis Testing Interpretations and Rényi Differential Privacy. AISTATS.
6. Bonawitz, K. et al. (2017). Practical Secure Aggregation for Privacy-Preserving Machine Learning. ACM CCS.
7. Yin, D. et al. (2018). Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates. ICML.
8. Dal Pozzolo, A. et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. IEEE SSCI.
9. Kairouz, P. et al. (2021). Advances and Open Problems in Federated Learning. Foundations and Trends in Machine Learning.

# Experiment summary

## Mode comparison (test set)

| Mode | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | eps | Uplink (MB) |
|---|---|---|---|---|---|---|---|---|
| A: Centralised | 1.000 | 0.588 | 0.740 | 0.807 | 0.977 | 0.0000 | - | 0.00 |
| B: Local-only | 0.963 | 0.645 | 0.772 | 0.803 | 0.974 | 0.0003 | - | 0.00 |
| C: FedAvg | 0.935 | 0.725 | 0.817 | 0.817 | 0.979 | 0.0005 | - | 1.25 |
| D: FedAvg + DP | 0.931 | 0.675 | 0.783 | 0.801 | 0.974 | 0.0005 | 7.29 | 1.25 |
| E: FedAvg + DP + SecAgg | 0.931 | 0.675 | 0.783 | 0.801 | 0.974 | 0.0005 | 7.29 | 2.50 |

## Privacy-utility sweep

| sigma | eps | PR-AUC | F1 | Recall |
|---|---|---|---|---|
| 0.5 | 80.20 | 0.814 | 0.782 | 0.650 |
| 0.8 | 15.31 | 0.808 | 0.779 | 0.662 |
| 1.1 | 7.29 | 0.801 | 0.783 | 0.675 |
| 1.5 | 4.37 | 0.791 | 0.777 | 0.675 |
| 2.0 | 2.91 | 0.774 | 0.765 | 0.650 |
| 3.0 | 1.75 | 0.742 | 0.729 | 0.637 |

## Poisoning robustness

| Attack | Aggregator | PR-AUC | F1 | Recall |
|---|---|---|---|---|
| none | fedavg | 0.817 | 0.817 | 0.725 |
| label_flip | fedavg | 0.010 | 0.020 | 1.000 |
| label_flip | fedavg + SecAgg | 0.010 | 0.020 | 1.000 |
| label_flip | median | 0.818 | 0.797 | 0.662 |
| label_flip | trimmed_mean | 0.842 | 0.788 | 0.650 |
| sign_flip | fedavg | 0.010 | 0.020 | 1.000 |
| sign_flip | fedavg + SecAgg | 0.013 | 0.025 | 1.000 |
| sign_flip | median | 0.811 | 0.786 | 0.688 |
| sign_flip | trimmed_mean | 0.816 | 0.769 | 0.625 |

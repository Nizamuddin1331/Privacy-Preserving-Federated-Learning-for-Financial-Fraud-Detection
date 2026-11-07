"""Publication-grade figures and a Markdown summary from saved result JSON files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import seaborn as sns  # noqa: E402

MODE_COLORS = {"A": "#1f2937", "B": "#9ca3af", "C": "#0f766e", "D": "#6d28d9", "E": "#b45309"}
sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)


def load_results(results_dir: str | Path) -> Dict[str, List[dict]]:
    out: Dict[str, List[dict]] = {}
    for f in sorted(Path(results_dir).glob("*.json")):
        r = json.loads(f.read_text())
        out.setdefault(r["config"]["tag"], []).append(r)
    return out


def _save(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return path


def plot_comparison(suite: List[dict], out: Path) -> Path:
    suite = sorted(suite, key=lambda r: r["config"]["mode"])
    metrics = ["precision", "recall", "f1", "pr_auc"]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    w = 0.8 / len(suite)
    for i, r in enumerate(suite):
        m = r["config"]["mode"]
        vals = [r["final"][k] for k in metrics]
        ax.bar(np.arange(len(metrics)) + i * w, vals, w, label=f"{m}: {r['mode_label']}", color=MODE_COLORS[m])
    ax.set_xticks(np.arange(len(metrics)) + w * (len(suite) - 1) / 2)
    ax.set_xticklabels(["Precision", "Recall", "F1", "PR-AUC"])
    ax.set_ylim(0, 1.05); ax.set_ylabel("Score (test set)")
    ax.set_title("Utility of each training mode")
    ax.legend(ncol=3, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    return _save(fig, out / "comparison.png")


def plot_convergence(suite: List[dict], out: Path) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for r in sorted(suite, key=lambda r: r["config"]["mode"]):
        m = r["config"]["mode"]; h = r["history"]
        axes[0].plot([x["round"] for x in h], [x["pr_auc"] for x in h], label=f"{m}: {r['mode_label']}",
                     color=MODE_COLORS[m], lw=2)
        if r.get("epsilon") is not None and m == "D":
            axes[1].plot([x["round"] for x in h], [x["epsilon"] for x in h], color=MODE_COLORS["D"], lw=2)
    axes[0].set(xlabel="Round (or local-epoch block)", ylabel="PR-AUC", title="Convergence")
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel="Round", ylabel=r"$\varepsilon$ at $\delta$=1e-5", title="Cumulative privacy budget (worst bank)")
    return _save(fig, out / "convergence.png")


def plot_confusion(suite: List[dict], out: Path) -> Path:
    suite = sorted(suite, key=lambda r: r["config"]["mode"])
    fig, axes = plt.subplots(1, len(suite), figsize=(3.2 * len(suite), 3.2))
    for ax, r in zip(np.atleast_1d(axes), suite):
        cm = np.array(r["confusion_matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues", ax=ax,
                    xticklabels=["Legit", "Fraud"], yticklabels=["Legit", "Fraud"])
        ax.set_title(f"{r['config']['mode']}: {r['mode_label']}", fontsize=9)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    return _save(fig, out / "confusion_matrices.png")


def plot_privacy_utility(sweep: List[dict], suite: List[dict], out: Path) -> Path:
    sweep = sorted(sweep, key=lambda r: r["epsilon"])
    eps = [r["epsilon"] for r in sweep]; pr = [r["final"]["pr_auc"] for r in sweep]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(eps, pr, "o-", color=MODE_COLORS["D"], lw=2, label="FedAvg + DP-SGD")
    for r in sweep:
        ax.annotate(f"σ={r['config']['noise_multiplier']}", (r["epsilon"], r["final"]["pr_auc"]),
                    textcoords="offset points", xytext=(4, -12), fontsize=8)
    for r in suite:
        if r["config"]["mode"] in ("A", "C"):
            ax.axhline(r["final"]["pr_auc"], ls="--", color=MODE_COLORS[r["config"]["mode"]],
                       label=f"{r['mode_label']} (no DP)")
    ax.set_xscale("log"); ax.set(xlabel=r"Privacy budget $\varepsilon$ (lower = stronger privacy)",
                                 ylabel="PR-AUC", title="Privacy-utility trade-off")
    ax.legend(fontsize=8)
    return _save(fig, out / "privacy_utility.png")


def plot_robustness(rob: List[dict], out: Path) -> Path:
    rows = {}
    for r in rob:
        c = r["config"]
        if c["n_malicious"] == 0:
            rows["clean"] = r["final"]["pr_auc"]; continue
        agg = c["aggregator"] + ("+secagg" if c["secure_aggregation"] else "")
        rows[(c["attack"], agg)] = r["final"]["pr_auc"]
    attacks = sorted({k[0] for k in rows if isinstance(k, tuple)})
    aggs = ["fedavg", "median", "trimmed_mean", "fedavg+secagg"]
    fig, ax = plt.subplots(figsize=(8, 4))
    w = 0.2
    for i, a in enumerate(aggs):
        ax.bar(np.arange(len(attacks)) + i * w, [rows.get((atk, a), 0) for atk in attacks], w, label=a)
    if "clean" in rows:
        ax.axhline(rows["clean"], ls="--", color="k", label="no attack")
    ax.set_xticks(np.arange(len(attacks)) + 1.5 * w); ax.set_xticklabels(attacks)
    ax.set(ylabel="PR-AUC", title="Poisoning robustness by aggregation rule"); ax.legend(fontsize=8)
    return _save(fig, out / "robustness.png")


def plot_partition(res: dict, out: Path) -> Path:
    p = res["partition"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar([x["client"] for x in p], [x["n"] for x in p], color="#0f766e")
    axes[0].set(xlabel="Bank", ylabel="Training records", title="Data volume per bank")
    axes[1].bar([x["client"] for x in p], [100 * x["fraud_rate"] for x in p], color="#b45309")
    axes[1].set(xlabel="Bank", ylabel="Fraud rate (%)", title="Fraud prevalence per bank")
    return _save(fig, out / "partition.png")


def summary_markdown(groups: Dict[str, List[dict]]) -> str:
    lines = ["# Experiment summary", ""]
    if "suite" in groups:
        lines += ["## Mode comparison (test set)", "",
                  "| Mode | Precision | Recall | F1 | PR-AUC | ROC-AUC | FPR | eps | Uplink (MB) |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for r in sorted(groups["suite"], key=lambda r: r["config"]["mode"]):
            f = r["final"]; eps = f"{r['epsilon']:.2f}" if r.get("epsilon") is not None else "-"
            lines.append(f"| {r['config']['mode']}: {r['mode_label']} | {f['precision']:.3f} | {f['recall']:.3f} | "
                         f"{f['f1']:.3f} | {f['pr_auc']:.3f} | {f['roc_auc']:.3f} | {f['fpr']:.4f} | {eps} | "
                         f"{r['uplink_bytes_total'] / 1e6:.2f} |")
        lines.append("")
    if "sweep" in groups:
        lines += ["## Privacy-utility sweep", "", "| sigma | eps | PR-AUC | F1 | Recall |", "|---|---|---|---|---|"]
        for r in sorted(groups["sweep"], key=lambda r: r["config"]["noise_multiplier"]):
            f = r["final"]
            lines.append(f"| {r['config']['noise_multiplier']} | {r['epsilon']:.2f} | {f['pr_auc']:.3f} | "
                         f"{f['f1']:.3f} | {f['recall']:.3f} |")
        lines.append("")
    if "robust" in groups:
        lines += ["## Poisoning robustness", "", "| Attack | Aggregator | PR-AUC | F1 | Recall |", "|---|---|---|---|---|"]
        for r in groups["robust"]:
            c, f = r["config"], r["final"]
            agg = c["aggregator"] + (" + SecAgg" if c["secure_aggregation"] else "")
            lines.append(f"| {c['attack'] if c['n_malicious'] else 'none'} | {agg} | {f['pr_auc']:.3f} | "
                         f"{f['f1']:.3f} | {f['recall']:.3f} |")
        lines.append("")
    return "\n".join(lines)


def generate_all(results_dir: str | Path = "results", reports_dir: str | Path = "reports") -> List[Path]:
    groups = load_results(results_dir)
    out = Path(reports_dir); out.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    if "suite" in groups:
        paths += [plot_comparison(groups["suite"], out), plot_convergence(groups["suite"], out),
                  plot_confusion(groups["suite"], out), plot_partition(groups["suite"][0], out)]
    if "sweep" in groups:
        paths.append(plot_privacy_utility(groups["sweep"], groups.get("suite", []), out))
    if "robust" in groups:
        paths.append(plot_robustness(groups["robust"], out))
    md = out / "summary.md"; md.write_text(summary_markdown(groups)); paths.append(md)
    return paths

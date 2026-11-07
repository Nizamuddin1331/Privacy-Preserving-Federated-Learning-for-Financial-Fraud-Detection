from typing import List, Optional

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.evaluation.visualizer import summary_markdown

from ..db.models import Experiment
from ..db.session import get_db

router = APIRouter(prefix="/metrics", tags=["metrics"])


def _done(db: Session, tag: Optional[str] = None) -> List[Experiment]:
    q = select(Experiment).where(Experiment.status == "completed").order_by(Experiment.id)
    if tag:
        q = q.where(Experiment.tag == tag)
    return list(db.scalars(q))


@router.get("/comparison")
def comparison(tag: str = "suite", db: Session = Depends(get_db)):
    """Latest completed run per mode (A-E)."""
    latest = {}
    for e in _done(db, tag):
        latest[e.mode] = e
    rows = []
    for m in sorted(latest):
        e = latest[m]; r = e.result
        rows.append({"id": e.id, "mode": m, "label": r["mode_label"], **{k: r["final"][k] for k in (
            "precision", "recall", "f1", "pr_auc", "roc_auc", "fpr", "fnr")},
            "epsilon": r.get("epsilon"), "uplink_mb": r["uplink_bytes_total"] / 1e6,
            "runtime_s": r["runtime_seconds"], "confusion_matrix": r["confusion_matrix"]})
    return rows


@router.get("/convergence")
def convergence(ids: str = Query("", description="comma-separated experiment ids; default = latest suite"),
                db: Session = Depends(get_db)):
    if ids:
        sel = [db.get(Experiment, int(i)) for i in ids.split(",") if i.strip().isdigit()]
        sel = [e for e in sel if e and e.status == "completed"]
    else:
        latest = {}
        for e in _done(db, "suite"):
            latest[e.mode] = e
        sel = [latest[m] for m in sorted(latest)]
    return [{"id": e.id, "mode": e.mode, "label": e.result["mode_label"], "name": e.name,
             "history": [{k: h.get(k) for k in ("round", "pr_auc", "f1", "recall", "precision", "train_loss", "epsilon")}
                         for h in e.result["history"]]} for e in sel]


@router.get("/privacy")
def privacy(db: Session = Depends(get_db)):
    rows = []
    for e in _done(db):
        r = e.result
        if r["config"]["dp"]:
            rows.append({"id": e.id, "name": e.name, "tag": e.tag, "mode": e.mode,
                         "noise_multiplier": r["config"]["noise_multiplier"], "epsilon": r["epsilon"],
                         "delta": r["config"]["delta"], "max_grad_norm": r["config"]["max_grad_norm"],
                         "pr_auc": r["final"]["pr_auc"], "f1": r["final"]["f1"], "recall": r["final"]["recall"]})
    return sorted(rows, key=lambda x: x["epsilon"])


@router.get("/secure-aggregation")
def secure_aggregation(db: Session = Depends(get_db)):
    rows = []
    for e in _done(db):
        r = e.result
        rows.append({"id": e.id, "name": e.name, "tag": e.tag, "mode": e.mode,
                     "secure": r["config"]["secure_aggregation"], "dp": r["config"]["dp"],
                     "dropout_prob": r["config"]["dropout_prob"],
                     "pr_auc": r["final"]["pr_auc"], "uplink_mb": r["uplink_bytes_total"] / 1e6,
                     "runtime_s": r["runtime_seconds"],
                     "dropped_total": sum(h.get("n_dropped") or 0 for h in r["history"])})
    return [x for x in rows if x["mode"] in ("C", "D", "E")]


@router.get("/robustness")
def robustness(db: Session = Depends(get_db)):
    rows = []
    for e in _done(db, "robust"):
        c, f = e.result["config"], e.result["final"]
        rows.append({"id": e.id, "attack": c["attack"] if c["n_malicious"] else "none",
                     "aggregator": c["aggregator"], "secure": c["secure_aggregation"],
                     "n_malicious": c["n_malicious"], "pr_auc": f["pr_auc"], "f1": f["f1"], "recall": f["recall"]})
    return rows


@router.get("/report", response_class=PlainTextResponse)
def report(db: Session = Depends(get_db)):
    groups = {}
    latest = {}
    for e in _done(db):
        key = (e.tag, e.name if e.tag != "suite" else e.mode)
        latest[key] = e.result
    for (tag, _), r in latest.items():
        groups.setdefault(tag, []).append(r)
    return summary_markdown(groups)

from typing import List, Optional

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import Session

from src.experiments.poisoning import robustness_configs
from src.experiments.runner import MODES, ExperimentConfig

from ..core.jobs import build_config, run_jobs
from ..db.models import Experiment
from ..db.session import get_db
from ..schemas.experiment import (ExperimentCreate, ExperimentOut, JobAccepted, RobustnessCreate,
                                  SweepCreate)

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _out(e: Experiment) -> ExperimentOut:
    o = ExperimentOut.model_validate(e)
    if e.result:
        o.final, o.epsilon = e.result["final"], e.result.get("epsilon")
    return o


def _validated(payload: dict) -> ExperimentConfig:
    try:
        return build_config(payload)
    except (ValueError, TypeError) as e:
        raise HTTPException(422, str(e))


def _add(db: Session, name: str, tag: str, cfg: ExperimentConfig) -> Experiment:
    e = Experiment(name=name, mode=cfg.mode, tag=tag, config=cfg.to_dict())
    db.add(e)
    db.flush()
    return e


@router.post("", response_model=JobAccepted, status_code=202)
def launch(body: ExperimentCreate, bg: BackgroundTasks, db: Session = Depends(get_db)):
    cfg = _validated(body.model_dump())
    if cfg.secure_aggregation and cfg.aggregator != "fedavg":
        raise HTTPException(422, "Robust aggregators are incompatible with secure aggregation")
    e = _add(db, body.name or f"{cfg.mode}_custom", "custom", cfg)
    db.commit()
    bg.add_task(run_jobs, [e.id])
    return JobAccepted(ids=[e.id], message="queued")


@router.post("/suite", response_model=JobAccepted, status_code=202)
def launch_suite(body: ExperimentCreate, bg: BackgroundTasks, db: Session = Depends(get_db)):
    """Run modes A-E with identical data / federation settings."""
    ids = []
    for m in MODES:
        p = {**body.model_dump(), "mode": m, "dp": None, "secure_aggregation": None,
             "n_malicious": 0, "attack": "none", "aggregator": "fedavg"}
        ids.append(_add(db, f"mode_{m}", "suite", _validated(p)).id)
    db.commit()
    bg.add_task(run_jobs, ids)
    return JobAccepted(ids=ids, message="suite queued (A-E)")


@router.post("/sweep", response_model=JobAccepted, status_code=202)
def launch_sweep(body: SweepCreate, bg: BackgroundTasks, db: Session = Depends(get_db)):
    ids = []
    for s in body.noise_multipliers:
        p = {**body.base.model_dump(), "mode": "D", "noise_multiplier": s, "dp": True, "secure_aggregation": False}
        ids.append(_add(db, f"sigma_{s}", "sweep", _validated(p)).id)
    db.commit()
    bg.add_task(run_jobs, ids)
    return JobAccepted(ids=ids, message="privacy sweep queued")


@router.post("/robustness", response_model=JobAccepted, status_code=202)
def launch_robustness(body: RobustnessCreate, bg: BackgroundTasks, db: Session = Depends(get_db)):
    base = _validated({**body.base.model_dump(), "mode": "C", "dp": False})
    cfgs = robustness_configs(base, attacks=body.attacks, n_malicious=body.n_malicious)
    ids = [_add(db, c.name, "robust", c).id for c in cfgs]
    db.commit()
    bg.add_task(run_jobs, ids)
    return JobAccepted(ids=ids, message="robustness grid queued")


@router.get("", response_model=List[ExperimentOut])
def list_experiments(tag: Optional[str] = None, status: Optional[str] = None, limit: int = 200,
                     db: Session = Depends(get_db)):
    q = select(Experiment).order_by(Experiment.id.desc()).limit(limit)
    if tag:
        q = q.where(Experiment.tag == tag)
    if status:
        q = q.where(Experiment.status == status)
    return [_out(e) for e in db.scalars(q)]


def _get(db: Session, eid: int) -> Experiment:
    e = db.get(Experiment, eid)
    if not e:
        raise HTTPException(404, f"Experiment {eid} not found")
    return e


@router.get("/{eid}", response_model=ExperimentOut)
def status(eid: int, db: Session = Depends(get_db)):
    return _out(_get(db, eid))


@router.get("/{eid}/results")
def results(eid: int, db: Session = Depends(get_db)):
    e = _get(db, eid)
    if e.status != "completed":
        raise HTTPException(409, f"Experiment is {e.status}")
    return e.result


@router.delete("/{eid}", status_code=204)
def delete(eid: int, db: Session = Depends(get_db)):
    db.delete(_get(db, eid))
    db.commit()

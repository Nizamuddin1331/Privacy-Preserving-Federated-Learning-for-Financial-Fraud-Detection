"""Background job execution. A global lock serialises runs (torch is CPU-hungry)."""
import threading
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.experiments.runner import ExperimentConfig, prepare_data, run_experiment, save_result

from ..db.models import Experiment
from ..db.session import SessionLocal
from .config import get_settings
from .logger import get_logger

log = get_logger()
_lock = threading.Lock()


def build_config(payload: dict) -> ExperimentConfig:
    s = get_settings()
    cfg = {k: v for k, v in payload.items() if v is not None or k in ("dp", "secure_aggregation")}
    cfg.pop("name", None)
    csv = Path(s.data_dir) / "creditcard.csv"
    cfg["dataset_path"] = str(csv) if csv.exists() else None
    return ExperimentConfig(**cfg)


def run_jobs(exp_ids: Iterable[int]) -> None:
    ids = list(exp_ids)
    with _lock:
        data_cache = {}
        for eid in ids:
            db = SessionLocal()
            try:
                exp = db.get(Experiment, eid)
                if exp is None or exp.status != "queued":
                    continue
                exp.status, exp.started_at = "running", datetime.now(timezone.utc)
                db.commit()
                try:
                    cfg = build_config({**exp.config, "name": exp.name, "tag": exp.tag})
                    cfg.name, cfg.tag = f"{exp.name}_{exp.id}", exp.tag
                    key = (cfg.dataset_path, cfg.n_synthetic, cfg.fraud_rate, cfg.seed)
                    if key not in data_cache:
                        data_cache[key] = prepare_data(cfg)

                    def progress(r, total):
                        exp.progress = r / total
                        db.commit()

                    res = run_experiment(cfg, data_cache[key], progress)
                    save_result(res, get_settings().results_dir)
                    exp.result, exp.status, exp.progress = res, "completed", 1.0
                    exp.config = res["config"]
                except Exception as e:  # noqa: BLE001
                    log.error(f"job {eid} failed: {e}\n{traceback.format_exc()}")
                    db.rollback()
                    exp = db.get(Experiment, eid)
                    exp.status, exp.error = "failed", str(e)
                exp.finished_at = datetime.now(timezone.utc)
                db.commit()
            finally:
                db.close()

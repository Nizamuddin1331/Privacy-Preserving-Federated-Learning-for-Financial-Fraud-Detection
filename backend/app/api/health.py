import platform
import time

import torch
from fastapi import APIRouter, Depends
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from ..db.models import Experiment
from ..db.session import get_db

router = APIRouter(tags=["health"])
_START = time.time()


@router.get("/health")
def health(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    counts = dict(db.execute(select(Experiment.status, func.count()).group_by(Experiment.status)).all())
    return {"status": "ok", "uptime_s": round(time.time() - _START, 1), "python": platform.python_version(),
            "torch": torch.__version__, "database": db.bind.dialect.name, "experiments": counts}

"""FastAPI entrypoint.  Run:  uvicorn app.main:app --reload   (from backend/)"""
import sys
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))                      # makes the `src` package importable

from fastapi import FastAPI  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402

from .api import dataset, experiments, health, metrics  # noqa: E402
from .core.config import get_settings  # noqa: E402
from .core.logger import get_logger  # noqa: E402
from .db.session import init_db  # noqa: E402

settings = get_settings()
log = get_logger(level=settings.log_level)


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    log.info("database ready")
    yield


app = FastAPI(title=settings.app_name, version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_methods=["*"], allow_headers=["*"])
for r in (health.router, dataset.router, experiments.router, metrics.router):
    app.include_router(r, prefix="/api")

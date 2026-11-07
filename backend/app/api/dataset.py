from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

from src.data.loader import (DatasetValidationError, dataset_summary, generate_synthetic, load_dataset,
                             validate_creditcard_df)
from src.data.partitioner import describe_partition, partition
from src.data.preprocessor import prepare

from ..core.config import get_settings
from ..schemas.experiment import SyntheticRequest

router = APIRouter(prefix="/dataset", tags=["dataset"])


def _csv() -> Path:
    return Path(get_settings().data_dir) / "creditcard.csv"


@router.get("/info")
def info(n_synthetic: int = Query(20_000, ge=2_000, le=300_000), fraud_rate: float = Query(0.01, gt=0.0005, lt=0.2)):
    """Summary of creditcard.csv if present, otherwise of the synthetic stand-in."""
    try:
        df, source = load_dataset(_csv(), n_synthetic, fraud_rate)
    except DatasetValidationError as e:
        raise HTTPException(422, str(e))
    return {**dataset_summary(df, source), "csv_present": _csv().exists(),
            "expected_path": str(_csv()), "head": df.head(8).round(3).to_dict(orient="records")}


@router.post("/synthetic")
def synthetic(req: SyntheticRequest):
    df = generate_synthetic(req.n_samples, req.fraud_rate, req.seed)
    out = {**dataset_summary(df, "synthetic")}
    if req.save:
        path = Path(get_settings().data_dir) / "synthetic_creditcard.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        out["saved_to"] = str(path)
    return out


@router.get("/partition-preview")
def partition_preview(n_clients: int = Query(5, ge=2, le=20), strategy: str = "iid", alpha: float = Query(0.5, gt=0.05),
                      n_synthetic: int = Query(20_000, ge=2_000, le=300_000), seed: int = 42):
    df, source = load_dataset(_csv(), n_synthetic, 0.01, seed)
    P = prepare(df, seed=seed)
    try:
        parts = partition(P.y_train, n_clients, strategy, alpha, seed)
    except (ValueError, RuntimeError) as e:
        raise HTTPException(422, str(e))
    return {"source": source, "strategy": strategy, "clients": describe_partition(parts, P.y_train)}

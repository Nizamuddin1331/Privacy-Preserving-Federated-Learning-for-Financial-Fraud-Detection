from datetime import datetime
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class ExperimentCreate(BaseModel):
    name: Optional[str] = Field(None, max_length=120)
    mode: Literal["A", "B", "C", "D", "E"] = "C"
    n_synthetic: int = Field(20_000, ge=2_000, le=300_000)
    fraud_rate: float = Field(0.01, gt=0.0005, lt=0.2)
    seed: int = 42
    n_clients: int = Field(5, ge=2, le=20)
    partition: Literal["iid", "dirichlet", "prevalence"] = "iid"
    dirichlet_alpha: float = Field(0.5, gt=0.05, le=100)
    rounds: int = Field(10, ge=1, le=100)
    local_epochs: int = Field(2, ge=1, le=10)
    batch_size: int = Field(256, ge=16, le=4096)
    lr: float = Field(0.05, gt=0, le=5)
    dp_lr: float = Field(0.5, gt=0, le=5)
    dropout_prob: float = Field(0.0, ge=0, lt=1)
    noise_multiplier: float = Field(1.1, gt=0.1, le=20)
    max_grad_norm: float = Field(2.0, gt=0, le=100)
    delta: float = Field(1e-5, gt=0, lt=0.01)
    dp: Optional[bool] = None
    secure_aggregation: Optional[bool] = None
    aggregator: Literal["fedavg", "median", "trimmed_mean"] = "fedavg"
    n_malicious: int = Field(0, ge=0, le=10)
    attack: Literal["none", "label_flip", "sign_flip", "gaussian"] = "none"
    attack_scale: float = Field(5.0, gt=0, le=100)


class SweepCreate(BaseModel):
    base: ExperimentCreate = ExperimentCreate()
    noise_multipliers: List[float] = Field([0.6, 0.9, 1.1, 1.6, 2.5], min_length=2, max_length=10)


class RobustnessCreate(BaseModel):
    base: ExperimentCreate = ExperimentCreate()
    attacks: List[Literal["label_flip", "sign_flip", "gaussian"]] = ["sign_flip", "label_flip"]
    n_malicious: int = Field(1, ge=1, le=5)


class ExperimentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: int
    name: str
    mode: str
    tag: str
    status: str
    progress: float
    config: Dict[str, Any]
    error: Optional[str] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    final: Optional[Dict[str, Any]] = None
    epsilon: Optional[float] = None


class JobAccepted(BaseModel):
    ids: List[int]
    message: str


class SyntheticRequest(BaseModel):
    n_samples: int = Field(40_000, ge=2_000, le=300_000)
    fraud_rate: float = Field(0.01, gt=0.0005, lt=0.2)
    seed: int = 42
    save: bool = False

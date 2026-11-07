from datetime import datetime, timezone

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from .session import Base


def _now():
    return datetime.now(timezone.utc)


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(120))
    mode: Mapped[str] = mapped_column(String(1), index=True)
    tag: Mapped[str] = mapped_column(String(32), default="custom", index=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)  # queued|running|completed|failed
    progress: Mapped[float] = mapped_column(Float, default=0.0)
    config: Mapped[dict] = mapped_column(JSON)
    result: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_now)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import Float, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from api.database import Base


class ModelVersion(Base):
    __tablename__ = "model_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    classifier_type: Mapped[str] = mapped_column(String(64), nullable=False)
    detector_type: Mapped[str] = mapped_column(String(64), nullable=False)
    # status lifecycle: training → active | failed; active → archived
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="training")
    path: Mapped[str | None] = mapped_column(String(512), nullable=True)
    metrics_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, default=lambda: datetime.now(timezone.utc))


class Alert(Base):
    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(primary_key=True)
    flow_id: Mapped[str] = mapped_column(String(64), nullable=False)
    # I2: append-only — only analyst_verdict and feedback_at may be updated
    consensus: Mapped[str] = mapped_column(String(32), nullable=False)
    severity: Mapped[str] = mapped_column(String(32), nullable=False)
    predicted_label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    classifier_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ae_score: Mapped[float] = mapped_column(Float, nullable=False)
    ae_threshold: Mapped[float] = mapped_column(Float, nullable=False)
    top_features_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    # Full feature vector stored for HITL replay buffer (fine_tune uses confirmed-benign rows)
    features_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    analyst_verdict: Mapped[str | None] = mapped_column(String(32), nullable=True)
    feedback_at: Mapped[datetime | None] = mapped_column(nullable=True)
    timestamp: Mapped[datetime] = mapped_column(
        nullable=False, default=lambda: datetime.now(timezone.utc)
    )

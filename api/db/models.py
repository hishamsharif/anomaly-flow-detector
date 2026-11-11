from __future__ import annotations

from datetime import datetime, timezone
from sqlalchemy import String, Text
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

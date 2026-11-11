from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, model_validator


class ModelVersionResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    classifier_type: str
    detector_type: str
    status: str
    metrics: dict[str, Any] | None = None
    created_at: datetime

    @model_validator(mode="before")
    @classmethod
    def _coerce_from_orm(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return v
        if hasattr(v, "metrics_json"):
            raw = getattr(v, "metrics_json", None)
            return {
                "id": v.id,
                "name": v.name,
                "classifier_type": v.classifier_type,
                "detector_type": v.detector_type,
                "status": v.status,
                "metrics": json.loads(raw) if raw else None,
                "created_at": v.created_at,
            }
        return v


class PaginatedModels(BaseModel):
    items: list[ModelVersionResponse]
    total: int


class TypeEntry(BaseModel):
    id: str
    label: str


class AvailableTypesResponse(BaseModel):
    classifiers: list[TypeEntry]
    detectors: list[TypeEntry]


class TrainingStartRequest(BaseModel):
    classifier_type: str
    detector_type: str
    mode: str = "initial"


class TrainingJobResponse(BaseModel):
    run_id: str


class TrainingStatusResponse(BaseModel):
    run_id: str
    pct: int
    mode: str
    started_at: str

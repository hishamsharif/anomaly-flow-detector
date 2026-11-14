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


# ── Inference schemas ──────────────────────────────────────────────────────────


class FeatureContribution(BaseModel):
    name: str
    contribution: float


class AlertResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    flow_id: str
    consensus: str
    severity: str
    predicted_label: int | None = None
    classifier_confidence: float | None = None
    ae_score: float
    ae_threshold: float
    top_features: list[FeatureContribution] = []
    analyst_verdict: str | None = None
    feedback_at: datetime | None = None
    timestamp: datetime

    @model_validator(mode="before")
    @classmethod
    def _coerce_from_orm(cls, v: Any) -> Any:
        if isinstance(v, dict):
            return v
        if hasattr(v, "top_features_json"):
            top_feats = json.loads(v.top_features_json) if v.top_features_json else []
            return {
                "id": v.id,
                "flow_id": v.flow_id,
                "consensus": v.consensus,
                "severity": v.severity,
                "predicted_label": v.predicted_label,
                "classifier_confidence": v.classifier_confidence,
                "ae_score": v.ae_score,
                "ae_threshold": v.ae_threshold,
                "top_features": top_feats,
                "analyst_verdict": v.analyst_verdict,
                "feedback_at": v.feedback_at,
                "timestamp": v.timestamp,
            }
        return v


class RejectedRow(BaseModel):
    row: int
    reason: str


class BatchSummary(BaseModel):
    total: int
    safe: int
    known_attack: int
    novel_attack: int
    rejected: int


class BatchPredictResponse(BaseModel):
    alerts: list[AlertResponse]
    rejected_rows: list[RejectedRow]
    summary: BatchSummary


class PredictJobResponse(BaseModel):
    job_id: str
    flow_count: int


class PredictJobStatusResponse(BaseModel):
    job_id: str
    pct: int
    done: bool


class PaginatedAlerts(BaseModel):
    items: list[AlertResponse]
    total: int
    page: int
    size: int


class VerdictRequest(BaseModel):
    verdict: str
    label: str | None = None


class VerdictCountResponse(BaseModel):
    eligible: int
    threshold: int
    unlocked: bool


# ── Dashboard schemas ──────────────────────────────────────────────────────────


class ConsensusCounts(BaseModel):
    safe: int = 0
    known_attack: int = 0
    novel_attack: int = 0


class SeverityCounts(BaseModel):
    low: int = 0
    medium: int = 0
    high: int = 0
    critical: int = 0


class DashboardSummary(BaseModel):
    # 24-hour windowed alert counts (timestamp set at insert time, not capture time)
    total_24h: int
    safe_24h: int
    known_attack_24h: int
    novel_attack_24h: int
    pending_verdict_24h: int  # alerts in last 24 h awaiting analyst verdict
    # Severity breakdown — all-time cumulative
    by_severity: SeverityCounts
    # HITL retrain eligibility — all-time cumulative running total
    verdict_eligible: int
    verdict_threshold: int
    retrain_unlocked: bool


# ── About schema ───────────────────────────────────────────────────────────────


class AboutResponse(BaseModel):
    version: str
    active_model: ModelVersionResponse | None
    config: dict[str, Any]

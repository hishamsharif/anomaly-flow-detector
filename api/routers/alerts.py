from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import get_settings
from api.database import get_db
from api.db import crud
from api.schemas import AlertResponse, PaginatedAlerts, VerdictCountResponse, VerdictRequest

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=PaginatedAlerts)
async def list_alerts(
    page: int = 1,
    size: int = 50,
    consensus: str | None = None,
    verdict: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    """
    GET /api/alerts

    Paginated alert listing with optional filters.
    consensus — SAFE | KNOWN_ATTACK | NOVEL_ATTACK
    verdict   — confirmed | override | uncertain | escalate | null (not yet submitted)
    """
    if page < 1:
        page = 1
    size = min(max(size, 1), 200)
    alerts, total = await crud.get_alerts_paginated(db, page, size, consensus, verdict)
    return PaginatedAlerts(
        items=[AlertResponse.model_validate(a) for a in alerts],
        total=total,
        page=page,
        size=size,
    )


@router.post("/{alert_id}/verdict", response_model=AlertResponse)
async def submit_verdict(
    alert_id: int,
    body: VerdictRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    POST /api/alerts/{id}/verdict

    Records analyst verdict on an alert (I2: immutable after first submission).
    Valid verdicts: confirmed | override | uncertain | escalate
    """
    valid_verdicts = {"confirmed", "override", "uncertain", "escalate"}
    if body.verdict not in valid_verdicts:
        raise HTTPException(status_code=422, detail=f"verdict must be one of {sorted(valid_verdicts)}")
    if body.verdict == "override" and not body.label:
        raise HTTPException(status_code=422, detail="label is required when verdict = override")

    alert = await crud.get_alert_by_id(db, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    if alert.analyst_verdict is not None:
        raise HTTPException(status_code=409, detail="Verdict already set — immutable after first submission (I2)")

    alert.analyst_verdict = body.verdict
    alert.feedback_at = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(alert)
    return AlertResponse.model_validate(alert)


@router.get("/verdict-count", response_model=VerdictCountResponse)
async def verdict_count(db: AsyncSession = Depends(get_db)):
    """GET /api/alerts/verdict-count — eligible verdict count vs retrain threshold."""
    settings = get_settings()
    eligible = await crud.get_eligible_verdict_count(db)
    threshold = settings.fine_tune_k_threshold
    return VerdictCountResponse(
        eligible=eligible,
        threshold=threshold,
        unlocked=eligible >= threshold,
    )

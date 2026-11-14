from __future__ import annotations

from datetime import UTC, datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.http.dao import alerts as alerts_dao
from api.http.dao.db_conn import get_db
from api.http.schemas import AlertResponse, PaginatedAlerts, VerdictCountResponse, VerdictRequest
from config.settings import get_settings

router = APIRouter(prefix="/api/alerts", tags=["alerts"])


@router.get("", response_model=PaginatedAlerts)
async def list_alerts(
    page: int = 1,
    size: int = 50,
    consensus: str | None = None,
    verdict: str | None = None,
    db: AsyncSession = Depends(get_db),
):
    if page < 1:
        page = 1
    size = min(max(size, 1), 200)
    alert_list, total = await alerts_dao.get_alerts_paginated(db, page, size, consensus, verdict)
    return PaginatedAlerts(
        items=[AlertResponse.model_validate(a) for a in alert_list],
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
    valid_verdicts = {"confirmed", "override", "uncertain", "escalate"}
    if body.verdict not in valid_verdicts:
        raise HTTPException(
            status_code=422, detail=f"verdict must be one of {sorted(valid_verdicts)}"
        )
    if body.verdict == "override" and not body.label:
        raise HTTPException(status_code=422, detail="label is required when verdict = override")

    alert = await alerts_dao.get_alert_by_id(db, alert_id)
    if alert is None:
        raise HTTPException(status_code=404, detail="Alert not found")
    if alert.analyst_verdict is not None:
        raise HTTPException(
            status_code=409, detail="Verdict already set — immutable after first submission (I2)"
        )

    alert.analyst_verdict = body.verdict
    alert.feedback_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(alert)
    return AlertResponse.model_validate(alert)


@router.get("/verdict-count", response_model=VerdictCountResponse)
async def verdict_count(db: AsyncSession = Depends(get_db)):
    settings = get_settings()
    eligible = await alerts_dao.get_eligible_verdict_count(db)
    threshold = settings.fine_tune_k_threshold
    return VerdictCountResponse(
        eligible=eligible,
        threshold=threshold,
        unlocked=eligible >= threshold,
    )

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.config import get_settings
from api.database import get_db
from api.db import crud
from api.schemas import DashboardSummary

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
async def dashboard_summary(db: AsyncSession = Depends(get_db)):
    """GET /api/dashboard/summary — alert counts by consensus + severity + retrain eligibility."""
    settings = get_settings()
    data = await crud.get_dashboard_summary(db, settings.fine_tune_k_threshold)
    return DashboardSummary(**data)

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.http.dao import dashboard as dashboard_dao
from api.http.dao.db_conn import get_db
from api.http.schemas import DashboardSummary
from config.settings import get_settings

router = APIRouter(prefix="/api/dashboard", tags=["dashboard"])


@router.get("/summary", response_model=DashboardSummary)
async def dashboard_summary(db: AsyncSession = Depends(get_db)):
    settings = get_settings()
    data = await dashboard_dao.get_dashboard_summary(db, settings.fine_tune_k_threshold)
    return DashboardSummary(**data)

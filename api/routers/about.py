from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from api.dao.db_conn import get_db
from api.dao import model_versions as model_versions_dao
from api.schemas import AboutResponse, ModelVersionResponse

router = APIRouter(prefix="/api/about", tags=["about"])


@router.get("", response_model=AboutResponse)
async def get_about(db: AsyncSession = Depends(get_db)) -> AboutResponse:
    settings = get_settings()
    active = await model_versions_dao.get_active_model(db)
    active_response = ModelVersionResponse.model_validate(active) if active else None

    config = {
        "fine_tune_k_threshold": settings.fine_tune_k_threshold,
        "ae_anomaly_threshold_percentile": settings.ae_anomaly_threshold_percentile,
        "ae_replay_buffer_fraction": settings.ae_replay_buffer_fraction,
        "ae_freeze_epochs": settings.ae_freeze_epochs,
        "device": settings.device,
        "far_gate_max": settings.far_gate_max,
        "recall_gate_max": settings.recall_gate_max,
        "clf_confidence_threshold": settings.clf_confidence_threshold,
        "sync_max_rows": settings.sync_max_rows,
        "max_upload_mb": settings.max_upload_mb,
        "variance_threshold": settings.variance_threshold,
        "correlation_threshold": settings.correlation_threshold,
        "feature_selection_k": settings.feature_selection_k,
        "feature_selection_method": settings.feature_selection_method,
    }

    return AboutResponse(
        version="2.0.0",
        active_model=active_response,
        config={k: str(v) for k, v in config.items()},
    )

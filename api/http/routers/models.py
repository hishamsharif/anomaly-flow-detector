from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from api.http.dao.db_conn import get_db
from api.http.dao import model_versions as model_versions_dao
from api.http.schemas import AvailableTypesResponse, ModelVersionResponse, PaginatedModels, TypeEntry
from pipeline.models.registry import CLASSIFIER_LABELS, DETECTOR_LABELS
from pipeline.store import ModelStore

router = APIRouter(prefix="/api/models", tags=["models"])


# Fixed-path routes must come before parameterised routes

@router.get("/available-types", response_model=AvailableTypesResponse)
async def available_types() -> AvailableTypesResponse:
    return AvailableTypesResponse(
        classifiers=[TypeEntry(id=k, label=v) for k, v in CLASSIFIER_LABELS.items()],
        detectors=[TypeEntry(id=k, label=v) for k, v in DETECTOR_LABELS.items()],
    )


@router.get("/active", response_model=ModelVersionResponse | None)
async def get_active(db: AsyncSession = Depends(get_db)) -> ModelVersionResponse | None:
    model = await model_versions_dao.get_active_model(db)
    if model is None:
        return None
    return ModelVersionResponse.model_validate(model)


@router.get("", response_model=PaginatedModels)
async def list_models(db: AsyncSession = Depends(get_db)) -> PaginatedModels:
    versions = await model_versions_dao.get_model_versions(db)
    items = [ModelVersionResponse.model_validate(v) for v in versions]
    return PaginatedModels(items=items, total=len(items))


@router.post("/{model_id}/promote", response_model=ModelVersionResponse)
async def promote_model(model_id: int, db: AsyncSession = Depends(get_db)) -> ModelVersionResponse:
    model = await model_versions_dao.get_model_version_by_id(db, model_id)
    if model is None:
        raise HTTPException(status_code=404, detail="Model not found")
    if model.status == "active":
        raise HTTPException(status_code=409, detail="Already active")
    if model.status == "training":
        raise HTTPException(status_code=409, detail="Cannot promote a model that is still training")
    if model.path is None:
        raise HTTPException(status_code=409, detail="Model has no saved artifact path")

    await model_versions_dao.async_set_active(db, model_id)

    try:
        ModelStore.get().load_into_memory(model.path)
    except Exception:
        pass

    updated = await model_versions_dao.get_model_version_by_id(db, model_id)
    return ModelVersionResponse.model_validate(updated)

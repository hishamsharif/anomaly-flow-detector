from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from api.database import get_db
from api.db import crud
from api.schemas import AvailableTypesResponse, ModelVersionResponse, PaginatedModels, TypeEntry
from core.registry import CLASSIFIER_LABELS, DETECTOR_LABELS

router = APIRouter(prefix="/api/models", tags=["models"])


# Fixed-path routes must come before any parameterised routes (e.g. /{id}/promote in Slice 5)

@router.get("/available-types", response_model=AvailableTypesResponse)
async def available_types() -> AvailableTypesResponse:
    return AvailableTypesResponse(
        classifiers=[TypeEntry(id=k, label=v) for k, v in CLASSIFIER_LABELS.items()],
        detectors=[TypeEntry(id=k, label=v) for k, v in DETECTOR_LABELS.items()],
    )


@router.get("/active", response_model=ModelVersionResponse | None)
async def get_active(db: AsyncSession = Depends(get_db)) -> ModelVersionResponse | None:
    model = await crud.get_active_model(db)
    if model is None:
        return None
    return ModelVersionResponse.model_validate(model)


@router.get("", response_model=PaginatedModels)
async def list_models(db: AsyncSession = Depends(get_db)) -> PaginatedModels:
    versions = await crud.get_model_versions(db)
    items = [ModelVersionResponse.model_validate(v) for v in versions]
    return PaginatedModels(items=items, total=len(items))

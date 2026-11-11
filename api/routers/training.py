from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from api import jobs
from api.config import get_settings
from api.database import get_db
from api.db import crud
from api.db.models import ModelVersion
from api.schemas import TrainingJobResponse, TrainingStartRequest, TrainingStatusResponse
from core.registry import CLASSIFIER_LABELS, CLASSIFIERS, DETECTOR_LABELS, DETECTORS
from training.pipelines.initial import run_initial_training

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start", status_code=202, response_model=TrainingJobResponse)
async def start_training(
    body: TrainingStartRequest,
    db: AsyncSession = Depends(get_db),
):
    if body.classifier_type not in CLASSIFIERS:
        raise HTTPException(status_code=422, detail=f"Unknown classifier_type: {body.classifier_type}")
    if body.detector_type not in DETECTORS:
        raise HTTPException(status_code=422, detail=f"Unknown detector_type: {body.detector_type}")
    if jobs.is_training():
        raise HTTPException(status_code=409, detail="A training job is already running")

    settings = get_settings()
    run_id = str(uuid4())

    version = ModelVersion(
        name="(pending)",
        classifier_type=body.classifier_type,
        detector_type=body.detector_type,
        status="training",
    )
    db.add(version)
    await db.commit()
    await db.refresh(version)

    clf_label = CLASSIFIER_LABELS.get(body.classifier_type, body.classifier_type)
    det_label = DETECTOR_LABELS.get(body.detector_type, body.detector_type)
    version_name = f"{clf_label} + {det_label} — v{version.id}"
    version.name = version_name
    await db.commit()

    db_url = settings.database_url

    def _emit(event: str, data: str) -> None:
        jobs.emit(run_id, event, data)

    jobs.launch(
        run_id,
        body.mode,
        run_initial_training,
        run_id,
        version.id,
        version_name,
        body.classifier_type,
        body.detector_type,
        settings,
        _emit,
        db_url,
    )

    return TrainingJobResponse(run_id=run_id)


@router.get("/{run_id}/events")
async def training_events(run_id: str):
    if jobs.get_status(run_id) is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return StreamingResponse(
        jobs.subscribe(run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/{run_id}/status", response_model=TrainingStatusResponse)
async def training_status(run_id: str):
    status = jobs.get_status(run_id)
    if status is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    return TrainingStatusResponse(**status)

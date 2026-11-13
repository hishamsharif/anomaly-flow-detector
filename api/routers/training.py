from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from api import jobs
from api.dao.db_conn import get_db
from runtime.manifest import get_host_environment
from api.dao import alerts as alerts_dao
from api.dao import model_versions as model_versions_dao
from api.schemas import TrainingJobResponse, TrainingStartRequest, TrainingStatusResponse
from core.registry import ANOMALY_DETECTOR_REGISTRY, CLASSIFIER_LABELS, CLASSIFIER_REGISTRY, DETECTOR_LABELS
from db.models import ModelVersion
from training.pipelines.fine_tune import run_ae_fine_tune
from training.pipelines.initial import run_initial_training

router = APIRouter(prefix="/api/training", tags=["training"])


@router.post("/start", status_code=202, response_model=TrainingJobResponse)
async def start_training(
    body: TrainingStartRequest,
    db: AsyncSession = Depends(get_db),
):
    if body.classifier_type not in CLASSIFIER_REGISTRY:
        raise HTTPException(status_code=422, detail=f"Unknown classifier_type: {body.classifier_type}")
    if body.detector_type not in ANOMALY_DETECTOR_REGISTRY:
        raise HTTPException(status_code=422, detail=f"Unknown detector_type: {body.detector_type}")
    if jobs.is_training():
        raise HTTPException(status_code=409, detail="A training job is already running")

    settings = get_settings()
    # Resolve device here (ADR-23): training/ must never call torch.cuda.is_available().
    # runtime.manifest caches the CUDA probe result from API startup.
    env = get_host_environment()
    device = "cuda" if (settings.device == "auto" and env.cuda_available) or settings.device == "cuda" else "cpu"

    prev_path: str | None = None
    if body.mode == "ae_retrain":
        active = await model_versions_dao.get_active_model(db)
        if active is None:
            raise HTTPException(status_code=409, detail="No active model — run initial training first")
        eligible = await alerts_dao.get_eligible_verdict_count(db)
        if eligible < settings.fine_tune_k_threshold:
            raise HTTPException(
                status_code=409,
                detail=f"Insufficient verdicts: {eligible} of {settings.fine_tune_k_threshold} required",
            )
        prev_path = active.path

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

    if body.mode == "ae_retrain":
        # Fetch analyst-confirmed benign flows in the router (async context) before handing
        # off to the background thread — keeps training/ free of api/dao imports (SPEC §9).
        analyst_flows = await alerts_dao.async_get_confirmed_benign_features(db)
        jobs.launch_training_job(
            run_id, body.mode, run_ae_fine_tune,
            run_id, version.id, version_name,
            body.classifier_type, body.detector_type,
            prev_path, settings, _emit, db_url, analyst_flows, device,
        )
    else:
        jobs.launch_training_job(
            run_id, body.mode, run_initial_training,
            run_id, version.id, version_name,
            body.classifier_type, body.detector_type,
            settings, _emit, db_url, device,
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

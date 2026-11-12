from __future__ import annotations

import json
from uuid import uuid4

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import get_settings
from api import jobs
from api.dao.db_conn import get_db
from api.dao import alerts as alerts_dao
from api.schemas import BatchPredictResponse, BatchSummary, PredictJobStatusResponse
from core.store import ModelStore
from inference import pipeline as infer_pipeline
from inference.adapters import csv_adapter, pcap_adapter

router = APIRouter(prefix="/api/predict", tags=["predict"])


def _flow_results_to_db_dicts(results, flows: list[dict], ae_threshold: float) -> list[dict]:
    return [
        {
            "flow_id": r.flow_id,
            "consensus": r.consensus,
            "severity": r.severity,
            "predicted_label": r.predicted_label,
            "classifier_confidence": r.classifier_confidence,
            "ae_score": r.ae_score,
            "ae_threshold": ae_threshold,
            "top_features_json": json.dumps(r.top_features),
            "features_json": json.dumps(flows[int(r.flow_id)]),
        }
        for r in results
    ]


def _build_batch_response(alert_list, rejected: list[dict]) -> BatchPredictResponse:
    from api.schemas import AlertResponse, RejectedRow

    safe = sum(1 for a in alert_list if a.consensus == "SAFE")
    known = sum(1 for a in alert_list if a.consensus == "KNOWN_ATTACK")
    novel = sum(1 for a in alert_list if a.consensus == "NOVEL_ATTACK")

    return BatchPredictResponse(
        alerts=[AlertResponse.model_validate(a) for a in alert_list],
        rejected_rows=[RejectedRow(**r) for r in rejected],
        summary=BatchSummary(
            total=len(alert_list) + len(rejected),
            safe=safe,
            known_attack=known,
            novel_attack=novel,
            rejected=len(rejected),
        ),
    )


def _run_async_inference(
    valid_rows: list[dict],
    rejected_rows: list[dict],
    bundle,
    clf_confidence_threshold: float,
    db_url: str,
) -> dict:
    """Background worker: run inference + sync DB insert, return serialisable result."""
    flow_results = infer_pipeline.run_batch_inference(
        valid_rows, bundle, clf_confidence_threshold=clf_confidence_threshold
    )
    db_dicts = _flow_results_to_db_dicts(flow_results, valid_rows, bundle.ae_threshold)
    alerts_dao.sync_insert_alerts(db_url, db_dicts)

    safe = sum(1 for r in flow_results if r.consensus == "SAFE")
    known = sum(1 for r in flow_results if r.consensus == "KNOWN_ATTACK")
    novel = sum(1 for r in flow_results if r.consensus == "NOVEL_ATTACK")

    return {
        "alerts": [
            {
                "flow_id": r.flow_id,
                "consensus": r.consensus,
                "severity": r.severity,
                "predicted_label": r.predicted_label,
                "classifier_confidence": r.classifier_confidence,
                "ae_score": r.ae_score,
                "ae_threshold": r.ae_threshold,
                "top_features": r.top_features,
            }
            for r in flow_results
        ],
        "rejected_rows": rejected_rows,
        "summary": {
            "total": len(flow_results) + len(rejected_rows),
            "safe": safe,
            "known_attack": known,
            "novel_attack": novel,
            "rejected": len(rejected_rows),
        },
    }


@router.post("/batch")
async def batch_predict(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    settings = get_settings()

    content = await file.read()
    if len(content) > settings.max_upload_mb * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"File exceeds {settings.max_upload_mb} MB limit")

    bundle = ModelStore.get().get_active()
    if bundle is None:
        raise HTTPException(status_code=409, detail="No active model — run training first")

    filename = (file.filename or "").lower()
    if filename.endswith((".pcap", ".cap")):
        valid_rows, rejected = pcap_adapter.parse_pcap(content, bundle.feature_names)
    elif filename.endswith(".csv"):
        valid_rows, rejected = csv_adapter.parse_csv(content, bundle.feature_names)
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type — upload a .csv or .pcap file")

    if not valid_rows:
        from api.schemas import RejectedRow
        return BatchPredictResponse(
            alerts=[],
            rejected_rows=[RejectedRow(**r) for r in rejected],
            summary=BatchSummary(total=len(rejected), safe=0, known_attack=0, novel_attack=0, rejected=len(rejected)),
        )

    if len(valid_rows) <= settings.sync_max_rows:
        flow_results = infer_pipeline.run_batch_inference(
            valid_rows, bundle, clf_confidence_threshold=settings.clf_confidence_threshold
        )
        db_dicts = _flow_results_to_db_dicts(flow_results, valid_rows, bundle.ae_threshold)
        inserted = await alerts_dao.bulk_insert_alerts(db, db_dicts)
        return _build_batch_response(inserted, rejected)

    else:
        job_id = str(uuid4())
        jobs.launch_predict_job(
            job_id,
            _run_async_inference,
            valid_rows,
            rejected,
            bundle,
            settings.clf_confidence_threshold,
            settings.database_url,
        )
        return JSONResponse(
            status_code=202,
            content={"job_id": job_id, "flow_count": len(valid_rows)},
        )


@router.get("/{job_id}/status", response_model=PredictJobStatusResponse)
async def predict_job_status(job_id: str):
    status = jobs.get_predict_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    return PredictJobStatusResponse(**status)


@router.get("/{job_id}/results")
async def predict_job_results(job_id: str):
    status = jobs.get_predict_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="job_id not found")
    if not status["done"]:
        raise HTTPException(status_code=409, detail="Job not complete yet")

    _, result = jobs.get_predict_result(job_id)
    if result is None:
        raise HTTPException(status_code=500, detail="Job failed — no result available")
    return result

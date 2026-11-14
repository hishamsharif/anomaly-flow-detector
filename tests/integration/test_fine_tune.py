"""Slice 5 tests — AE fine-tune pipeline + API endpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from httpx import AsyncClient

from db.models import Alert, ModelVersion

# asyncio_mode = "auto" in pyproject.toml handles async tests; no module-level mark needed.

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def low_threshold():
    """Temporarily set fine_tune_k_threshold to 2 for speed."""
    from config.settings import get_settings

    s = get_settings()
    old = s.fine_tune_k_threshold
    object.__setattr__(s, "fine_tune_k_threshold", 2)
    yield 2
    object.__setattr__(s, "fine_tune_k_threshold", old)


def _alert_dict(**kwargs) -> dict:
    defaults = dict(
        flow_id="0",
        consensus="SAFE",
        severity="low",
        ae_score=0.1,
        ae_threshold=0.9,
        top_features_json=json.dumps([]),
        analyst_verdict=None,
        timestamp=datetime.now(UTC),
    )
    defaults.update(kwargs)
    return defaults


async def _insert_alert(client: AsyncClient, **kwargs) -> Alert:
    from api.http.dao.db_conn import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        a = Alert(**_alert_dict(**kwargs))
        db.add(a)
        await db.commit()
        await db.refresh(a)
        return a


async def _insert_model(path: str | None = None, status: str = "active") -> int:
    from api.http.dao.db_conn import AsyncSessionLocal

    async with AsyncSessionLocal() as db:
        m = ModelVersion(
            name="v1",
            classifier_type="xgboost",
            detector_type="mlp_ae",
            status=status,
            path=path,
        )
        db.add(m)
        await db.commit()
        await db.refresh(m)
        return m.id


# ── Unit: MLPAutoencoder.fine_tune ────────────────────────────────────────────


def test_mlp_ae_fine_tune_runs():
    from pipeline.models.detector.autoencoder import MLPAutoencoder

    X = np.random.rand(80, 20).astype(np.float32)
    ae = MLPAutoencoder(device="cpu", epochs=4, batch_size=16)
    ae.fit(X)
    ae.calibrate_threshold(X)
    threshold_before = ae.threshold

    ae.fine_tune(X, freeze_epochs=2)

    scores = ae.score(X)
    assert scores.shape == (80,)
    assert (scores >= 0).all()
    assert ae.threshold == threshold_before  # threshold unchanged by fine_tune itself


def test_mlp_ae_fine_tune_save_load(tmp_path):
    from pipeline.models.detector.autoencoder import MLPAutoencoder

    X = np.random.rand(60, 10).astype(np.float32)
    ae = MLPAutoencoder(device="cpu", epochs=2, batch_size=16)
    ae.fit(X)
    ae.calibrate_threshold(X)
    ae.fine_tune(X, freeze_epochs=1)
    ae.save(str(tmp_path))

    ae2 = MLPAutoencoder()
    ae2.load(str(tmp_path))
    scores = ae2.score(X[:5])
    assert scores.shape == (5,)


# ── Unit: save_artifacts copy_clf_from ────────────────────────────────────────


def test_save_artifacts_copy_clf_from(tmp_path):
    from sklearn.preprocessing import StandardScaler

    from pipeline.models.detector.autoencoder import MLPAutoencoder
    from pipeline.preprocessing.feature_engineering import FeatureEngResult
    from pipeline.store import ModelStore

    # Create a fake classifier.joblib in a "previous" dir
    src = tmp_path / "prev"
    src.mkdir()
    (src / "classifier.joblib").write_bytes(b"fake-clf-bytes")

    ae = MLPAutoencoder(device="cpu", epochs=1, batch_size=16)
    ae.fit(np.random.rand(20, 5).astype(np.float32))
    ae.calibrate_threshold(np.random.rand(10, 5).astype(np.float32))

    feat_result = FeatureEngResult(
        variance_mask=np.ones(5, dtype=bool),
        corr_mask=np.ones(5, dtype=bool),
        selector_mask=np.ones(5, dtype=bool),
        feature_names=["f0", "f1", "f2", "f3", "f4"],
        classifier_scaler=StandardScaler(),
        detector_scaler=StandardScaler(),
    )

    store = ModelStore.get()
    artifact_path = store.save_artifacts(
        run_id="test-run",
        name="test_model",
        clf=None,
        ae=ae,
        feat_result=feat_result,
        metrics={"ae_threshold": 0.5, "novel_critical_threshold": 0.3},
        artifact_dir=str(tmp_path),
        copy_clf_from=str(src),
    )

    # Verify the classifier bytes were copied unchanged (I5)
    assert (Path(artifact_path) / "classifier.joblib").read_bytes() == b"fake-clf-bytes"
    # Verify detector.pt was saved by the AE
    assert (Path(artifact_path) / "detector.pt").exists()


# ── API: ae_retrain 409 guards ────────────────────────────────────────────────


async def test_ae_retrain_409_no_active_model(client: AsyncClient):
    res = await client.post(
        "/api/training/start",
        json={
            "classifier_type": "xgboost",
            "detector_type": "mlp_ae",
            "mode": "ae_retrain",
        },
    )
    assert res.status_code == 409
    assert "No active model" in res.json()["detail"]


async def test_ae_retrain_409_insufficient_verdicts(client: AsyncClient, low_threshold, tmp_path):
    await _insert_model(path=str(tmp_path), status="active")

    # Insert only 1 confirmed alert (below threshold of 2)
    await _insert_alert(client, analyst_verdict="confirmed")

    res = await client.post(
        "/api/training/start",
        json={
            "classifier_type": "xgboost",
            "detector_type": "mlp_ae",
            "mode": "ae_retrain",
        },
    )
    assert res.status_code == 409
    assert "Insufficient verdicts" in res.json()["detail"]


async def test_ae_retrain_202_when_unlocked(client: AsyncClient, low_threshold, tmp_path):
    await _insert_model(path=str(tmp_path), status="active")

    # Insert threshold (2) confirmed alerts
    await _insert_alert(client, analyst_verdict="confirmed")
    await _insert_alert(client, analyst_verdict="confirmed")

    with patch("api.http.routers.training.run_ae_fine_tune") as mock_fn:

        def _immediate(*args):
            # find emit callable — it's the second-to-last positional arg
            emit = args[-2]
            emit("step", "100")
            emit("done", "done")

        mock_fn.side_effect = _immediate

        res = await client.post(
            "/api/training/start",
            json={
                "classifier_type": "xgboost",
                "detector_type": "mlp_ae",
                "mode": "ae_retrain",
            },
        )

    assert res.status_code == 202
    assert "run_id" in res.json()


# ── API: promote endpoint ──────────────────────────────────────────────────────


async def test_promote_404(client: AsyncClient):
    res = await client.post("/api/models/99999/promote")
    assert res.status_code == 404


async def test_promote_409_already_active(client: AsyncClient, tmp_path):
    model_id = await _insert_model(path=str(tmp_path), status="active")
    res = await client.post(f"/api/models/{model_id}/promote")
    assert res.status_code == 409
    assert "Already active" in res.json()["detail"]


async def test_promote_409_training(client: AsyncClient):
    model_id = await _insert_model(path=None, status="training")
    res = await client.post(f"/api/models/{model_id}/promote")
    assert res.status_code == 409
    assert "still training" in res.json()["detail"]


async def test_promote_409_no_path(client: AsyncClient):
    model_id = await _insert_model(path=None, status="failed")
    res = await client.post(f"/api/models/{model_id}/promote")
    assert res.status_code == 409
    assert "artifact path" in res.json()["detail"]


async def test_promote_200_sets_active(client: AsyncClient, tmp_path):
    # Create an active model (will be archived on promote)
    await _insert_model(path=str(tmp_path / "old"), status="active")
    # Create a completed model to promote
    target_id = await _insert_model(path=str(tmp_path / "new"), status="failed")

    with patch("api.http.routers.models.ModelStore") as mock_store:
        mock_store.get.return_value.load_into_memory.return_value = None
        res = await client.post(f"/api/models/{target_id}/promote")

    assert res.status_code == 200
    body = res.json()
    assert body["id"] == target_id
    assert body["status"] == "active"

    # Previous active model should now be archived
    models_res = await client.get("/api/models")
    all_models = {m["id"]: m for m in models_res.json()["items"]}
    assert all_models[target_id]["status"] == "active"

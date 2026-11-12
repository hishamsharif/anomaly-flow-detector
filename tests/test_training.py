"""Slice 2 tests — component isolation + API endpoint shape."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from httpx import AsyncClient


# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_df(n: int = 200, attack_frac: float = 0.3, seed: int = 42) -> pd.DataFrame:
    fixture = Path(__file__).parent / "fixtures" / "sample_flows.csv"
    raw = pd.read_csv(fixture).head(n)
    raw.columns = [c.strip() for c in raw.columns]
    num_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
    raw = raw[num_cols + ["Label"]]
    raw["label"] = (raw["Label"] != "BENIGN").astype(int)
    return raw.drop(columns=["Label"])


def _make_settings(**overrides):
    from api.config import get_settings
    s = get_settings()
    for k, v in overrides.items():
        object.__setattr__(s, k, v)
    return s


# ── Feature engineering ───────────────────────────────────────────────────────

def test_variance_filter_reduces_features():
    from training.feature_eng.variance_filter import VarianceFilter
    X = np.hstack([np.random.rand(100, 5), np.zeros((100, 2))])
    vf = VarianceFilter(threshold=0.0)
    X_out = vf.fit_transform(X)
    assert X_out.shape[1] == 5
    assert vf.mask.sum() == 5


def test_correlation_filter():
    from training.feature_eng.correlation_filter import CorrelationFilter
    rng = np.random.default_rng(0)
    X = rng.random((100, 4))
    X[:, 1] = X[:, 0]  # perfectly correlated pair
    names = ["a", "b", "c", "d"]
    cf = CorrelationFilter(threshold=0.95)
    X_out = cf.fit_transform(X, names)
    assert X_out.shape[1] < 4
    assert cf.mask.sum() == X_out.shape[1]


def test_selector_passthrough_when_k0():
    from training.feature_eng.selector import FeatureSelector
    X = np.random.rand(50, 10)
    y = np.array([0] * 25 + [1] * 25)
    sel = FeatureSelector(k=0, method="mi")
    X_out = sel.fit_transform(X, y)
    assert X_out.shape == X.shape
    assert sel.mask.all()


def test_feature_eng_pipeline_runs():
    from api.config import get_settings
    df = _make_df(300)
    split = int(len(df) * 0.6)
    vsplit = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    val_df = df.iloc[split:vsplit].copy()
    test_df = df.iloc[vsplit:].copy()

    settings = get_settings()
    from training.feature_eng.pipeline import run_feature_engineering
    result = run_feature_engineering(train_df, val_df, test_df, settings)
    X_train_classifier = result[0]
    feat_res = result[-1]
    assert X_train_classifier.shape[0] > 0
    assert len(feat_res.feature_names) > 0


# ── XGBoost classifier ────────────────────────────────────────────────────────

def test_xgboost_fit_predict():
    from core.classifiers.xgboost_clf import XGBoostClassifier
    rng = np.random.default_rng(1)
    X = rng.random((200, 10)).astype(np.float32)
    y = np.array([0] * 100 + [1] * 100)
    clf = XGBoostClassifier(device="cpu")
    clf.fit(X, y)
    labels, confs = clf.predict(X[:10])
    assert labels.shape == (10,)
    assert confs.shape == (10,)
    assert confs.min() >= 0.0 and confs.max() <= 1.0


def test_xgboost_save_load(tmp_path):
    from core.classifiers.xgboost_clf import XGBoostClassifier
    X = np.random.rand(100, 5).astype(np.float32)
    y = np.array([0] * 50 + [1] * 50)
    clf = XGBoostClassifier(device="cpu")
    clf.fit(X, y)
    clf.save(str(tmp_path))
    clf2 = XGBoostClassifier()
    clf2.load(str(tmp_path))
    labels, _ = clf2.predict(X[:5])
    assert labels.shape == (5,)


# ── MLP Autoencoder ───────────────────────────────────────────────────────────

def test_mlp_ae_fit_score():
    from core.detectors.mlp_ae import MLPAutoencoder
    X = np.random.rand(100, 20).astype(np.float32)
    ae = MLPAutoencoder(device="cpu", epochs=2, batch_size=32)
    ae.fit(X)
    scores = ae.score(X)
    assert scores.shape == (100,)
    assert (scores >= 0).all()


def test_mlp_ae_calibrate_threshold():
    from core.detectors.mlp_ae import MLPAutoencoder
    X = np.random.rand(200, 20).astype(np.float32)
    ae = MLPAutoencoder(device="cpu", epochs=2, batch_size=32)
    ae.fit(X)
    threshold = ae.calibrate_threshold(X, percentile=95.0)
    assert threshold > 0
    assert ae.threshold == threshold


def test_mlp_ae_save_load(tmp_path):
    from core.detectors.mlp_ae import MLPAutoencoder
    X = np.random.rand(50, 10).astype(np.float32)
    ae = MLPAutoencoder(device="cpu", epochs=1, batch_size=16)
    ae.fit(X)
    ae.calibrate_threshold(X)
    ae.save(str(tmp_path))
    ae2 = MLPAutoencoder()
    ae2.load(str(tmp_path))
    scores = ae2.score(X[:5])
    assert scores.shape == (5,)


# ── API training endpoints ────────────────────────────────────────────────────

async def test_training_start_returns_202(client: AsyncClient):
    with patch("api.routers.training.run_initial_training") as mock_fn:
        # Make the training job complete immediately
        def _immediate(run_id, version_id, version_name, clf_type, det_type, settings, emit, db_url):
            emit("step", "100")
            emit("done", "done")
        mock_fn.side_effect = _immediate

        resp = await client.post("/api/training/start", json={
            "classifier_type": "xgboost",
            "detector_type": "mlp_ae",
        })
    assert resp.status_code == 202
    data = resp.json()
    assert "run_id" in data
    assert len(data["run_id"]) > 0


async def test_training_start_409_when_already_training(client: AsyncClient):
    import threading, time
    from api import jobs

    barrier = threading.Event()

    def _slow(run_id, version_id, version_name, clf_type, det_type, settings, emit, db_url):
        barrier.wait(timeout=5)
        emit("done", "done")

    with patch("api.routers.training.run_initial_training", side_effect=_slow):
        resp1 = await client.post("/api/training/start", json={
            "classifier_type": "xgboost",
            "detector_type": "mlp_ae",
        })
        assert resp1.status_code == 202

        # Second request while first is running
        resp2 = await client.post("/api/training/start", json={
            "classifier_type": "xgboost",
            "detector_type": "mlp_ae",
        })
        assert resp2.status_code == 409

    barrier.set()


async def test_training_status_404_unknown(client: AsyncClient):
    resp = await client.get("/api/training/nonexistent-id/status")
    assert resp.status_code == 404


async def test_training_status_200_after_start(client: AsyncClient):
    with patch("api.routers.training.run_initial_training") as mock_fn:
        done_event = __import__("threading").Event()

        def _slow(run_id, version_id, version_name, clf_type, det_type, settings, emit, db_url):
            done_event.wait(timeout=3)
            emit("done", "done")
        mock_fn.side_effect = _slow

        resp = await client.post("/api/training/start", json={
            "classifier_type": "xgboost",
            "detector_type": "mlp_ae",
        })
        assert resp.status_code == 202
        run_id = resp.json()["run_id"]

        status = await client.get(f"/api/training/{run_id}/status")
        assert status.status_code == 200
        body = status.json()
        assert body["run_id"] == run_id
        assert "pct" in body

        done_event.set()


async def test_training_start_422_unknown_classifier(client: AsyncClient):
    resp = await client.post("/api/training/start", json={
        "classifier_type": "unknown_clf",
        "detector_type": "mlp_ae",
    })
    assert resp.status_code == 422

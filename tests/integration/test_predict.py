"""Slice 3 tests — inference pipeline + predict/alerts endpoints."""
from __future__ import annotations

import io
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from httpx import AsyncClient


# ── Helpers ───────────────────────────────────────────────────────────────────

FEATURE_NAMES = ["feat_a", "feat_b", "feat_c", "feat_d", "feat_e"]
N_FEATURES = len(FEATURE_NAMES)


def _make_bundle(ae_threshold: float = 0.5, novel_critical_threshold: float = 0.3):
    """Return a mock ArtifactBundle that mimics the real object's interface."""
    from pipeline.store import ArtifactBundle
    from sklearn.preprocessing import StandardScaler
    import numpy as np

    clf = MagicMock()
    # label=1 (attack), confidence=0.9 → KNOWN_ATTACK; side_effect scales to input size
    clf.predict.side_effect = lambda X: (
        np.ones(len(X), dtype=int),
        np.full(len(X), 0.9),
    )
    clf.feature_importances = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

    ae = MagicMock()
    ae.score.side_effect = lambda X: np.full(len(X), 0.1)        # < ae_threshold (0.5)
    ae.score_per_feature.side_effect = lambda X: np.full((len(X), N_FEATURES), 0.01)

    scaler = StandardScaler()
    X_dummy = np.random.rand(10, N_FEATURES).astype(np.float32)
    scaler.fit(X_dummy)

    return ArtifactBundle(
        clf=clf,
        ae=ae,
        classifier_scaler=scaler,
        detector_scaler=scaler,
        feature_names=FEATURE_NAMES,
        variance_mask=np.ones(N_FEATURES, dtype=bool),
        corr_mask=np.ones(N_FEATURES, dtype=bool),
        selector_mask=np.ones(N_FEATURES, dtype=bool),
        ae_threshold=ae_threshold,
        novel_critical_threshold=novel_critical_threshold,
    )


def _make_csv_bytes(feature_names: list[str], n_rows: int = 3, attack: bool = False) -> bytes:
    rng = np.random.default_rng(42)
    data = {f: rng.random(n_rows) for f in feature_names}
    df = pd.DataFrame(data)
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    return buf.getvalue()


# ── csv_adapter unit tests ────────────────────────────────────────────────────

def test_csv_adapter_accepts_valid_rows():
    from pipeline.inference.adapters.csv_adapter import parse_csv
    content = _make_csv_bytes(FEATURE_NAMES, n_rows=5)
    valid, rejected = parse_csv(content, FEATURE_NAMES)
    assert len(valid) == 5
    assert len(rejected) == 0
    assert set(valid[0].keys()) == set(FEATURE_NAMES)


def test_csv_adapter_rejects_missing_feature():
    from pipeline.inference.adapters.csv_adapter import parse_csv
    # CSV missing feat_e
    partial_names = FEATURE_NAMES[:-1]
    content = _make_csv_bytes(partial_names, n_rows=3)
    valid, rejected = parse_csv(content, FEATURE_NAMES)
    assert len(valid) == 0
    assert len(rejected) == 3
    assert "missing feature" in rejected[0]["reason"]


def test_csv_adapter_rejects_nan_rows():
    from pipeline.inference.adapters.csv_adapter import parse_csv
    import io
    # Row 1 has NaN in feat_a
    df = pd.DataFrame({f: [1.0, 2.0, 3.0] for f in FEATURE_NAMES})
    df.at[1, "feat_a"] = float("nan")
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    valid, rejected = parse_csv(buf.getvalue(), FEATURE_NAMES)
    assert len(valid) == 2
    assert len(rejected) == 1
    assert "NaN" in rejected[0]["reason"]


# ── Inference pipeline unit tests ─────────────────────────────────────────────

def test_pipeline_known_attack_verdict():
    from pipeline.inference.predict_classifier import run_batch_inference
    bundle = _make_bundle()
    # clf returns label=1, conf=0.9 → KNOWN_ATTACK
    flows = [{f: 1.0 for f in FEATURE_NAMES}]
    results = run_batch_inference(flows, bundle, clf_confidence_threshold=0.5)
    assert len(results) == 1
    assert results[0].consensus == "KNOWN_ATTACK"
    assert results[0].severity in ("high", "medium")
    assert len(results[0].top_features) > 0


def test_pipeline_novel_attack_verdict():
    from pipeline.inference.predict_classifier import run_batch_inference
    bundle = _make_bundle(ae_threshold=0.05, novel_critical_threshold=0.03)
    # clf returns label=0 (benign), AE score > threshold → NOVEL_ATTACK
    bundle.clf.predict.side_effect = lambda X: (np.zeros(len(X), dtype=int), np.full(len(X), 0.4))
    bundle.ae.score.side_effect = lambda X: np.full(len(X), 0.1)  # > ae_threshold(0.05)
    flows = [{f: 1.0 for f in FEATURE_NAMES}]
    results = run_batch_inference(flows, bundle, clf_confidence_threshold=0.5)
    assert results[0].consensus == "NOVEL_ATTACK"
    assert results[0].severity in ("critical", "high")


def test_pipeline_safe_verdict():
    from pipeline.inference.predict_classifier import run_batch_inference
    bundle = _make_bundle(ae_threshold=0.5)
    # clf returns label=0 (benign), AE score < threshold → SAFE
    bundle.clf.predict.side_effect = lambda X: (np.zeros(len(X), dtype=int), np.full(len(X), 0.3))
    bundle.ae.score.side_effect = lambda X: np.full(len(X), 0.1)  # < ae_threshold(0.5)
    flows = [{f: 1.0 for f in FEATURE_NAMES}]
    results = run_batch_inference(flows, bundle, clf_confidence_threshold=0.5)
    assert results[0].consensus == "SAFE"
    assert results[0].severity == "low"


def test_pipeline_low_confidence_passes_to_stage3():
    """Stage 2 low-confidence flow passes to Stage 3 (confidence gate §5.2)."""
    from pipeline.inference.predict_classifier import run_batch_inference
    bundle = _make_bundle(ae_threshold=0.5)
    # clf returns label=1 (attack) but conf < threshold → BENIGN (passes to Stage 3)
    bundle.clf.predict.side_effect = lambda X: (np.ones(len(X), dtype=int), np.full(len(X), 0.3))
    bundle.ae.score.side_effect = lambda X: np.full(len(X), 0.1)  # < threshold → SAFE
    flows = [{f: 1.0 for f in FEATURE_NAMES}]
    results = run_batch_inference(flows, bundle, clf_confidence_threshold=0.5)
    # Stage 2 didn't fire (conf=0.3 < 0.5), Stage 3 ae_score=0.1 < 0.5 → SAFE
    assert results[0].consensus == "SAFE"


def test_pipeline_top_features_populated():
    from pipeline.inference.predict_classifier import run_batch_inference
    bundle = _make_bundle()
    flows = [{f: 1.0 for f in FEATURE_NAMES}]
    results = run_batch_inference(flows, bundle, clf_confidence_threshold=0.5)
    assert results[0].top_features
    assert all("name" in tf and "contribution" in tf for tf in results[0].top_features)


def test_pipeline_stage1_safe_rule():
    from pipeline.rule_engine import RuleEngine
    from pipeline.inference.predict_classifier import run_batch_inference
    bundle = _make_bundle()
    re = RuleEngine(port_verdict_rules={}, allowlisted_ports=frozenset({443}))
    flows = [{"Destination Port": 443.0, **{f: 1.0 for f in FEATURE_NAMES}}]
    results = run_batch_inference(flows, bundle, rule_engine=re)
    assert results[0].consensus == "SAFE"
    # Stage 1 exit: no classifier or AE call
    bundle.clf.predict.assert_not_called()


def test_pipeline_stage1_known_attack_rule():
    from pipeline.rule_engine import RuleEngine
    from pipeline.inference.predict_classifier import run_batch_inference
    bundle = _make_bundle()
    re = RuleEngine(port_verdict_rules={21: "KNOWN_ATTACK"}, allowlisted_ports=frozenset())
    flows = [{"Destination Port": 21.0, **{f: 1.0 for f in FEATURE_NAMES}}]
    results = run_batch_inference(flows, bundle, rule_engine=re)
    assert results[0].consensus == "KNOWN_ATTACK"
    bundle.clf.predict.assert_not_called()


# ── Severity derivation ───────────────────────────────────────────────────────

def test_severity_safe():
    from pipeline.inference.predict_classifier import _derive_severity
    assert _derive_severity("SAFE", None, 0.0, 0.5, 0.3) == "low"


def test_severity_known_attack_high():
    from pipeline.inference.predict_classifier import _derive_severity
    assert _derive_severity("KNOWN_ATTACK", 0.9, 0.0, 0.5, 0.3) == "high"


def test_severity_known_attack_medium():
    from pipeline.inference.predict_classifier import _derive_severity
    assert _derive_severity("KNOWN_ATTACK", 0.6, 0.0, 0.5, 0.3) == "medium"


def test_severity_novel_attack_critical():
    from pipeline.inference.predict_classifier import _derive_severity
    # ae_score (0.6) > novel_critical_threshold (0.3) → critical
    assert _derive_severity("NOVEL_ATTACK", None, 0.6, 0.5, 0.3) == "critical"


def test_severity_novel_attack_high():
    from pipeline.inference.predict_classifier import _derive_severity
    # ae_score (0.4) ≤ novel_critical_threshold (0.5) → high
    assert _derive_severity("NOVEL_ATTACK", None, 0.4, 0.3, 0.5) == "high"


# ── API endpoint tests ────────────────────────────────────────────────────────

async def test_batch_predict_409_no_active_model(client: AsyncClient):
    with patch("api.routers.predict.ModelStore") as mock_store:
        mock_store.get.return_value.get_active.return_value = None
        resp = await client.post(
            "/api/predict/batch",
            files={"file": ("flows.csv", b"a,b\n1,2\n", "text/csv")},
        )
    assert resp.status_code == 409


async def test_batch_predict_413_oversized(client: AsyncClient):
    with patch("api.routers.predict.ModelStore") as mock_store:
        mock_store.get.return_value.get_active.return_value = _make_bundle()
        big_content = b"x" * (200 * 1024 * 1024)
        resp = await client.post(
            "/api/predict/batch",
            files={"file": ("flows.csv", big_content, "text/csv")},
        )
    assert resp.status_code == 413


async def test_batch_predict_sync_returns_200(client: AsyncClient):
    bundle = _make_bundle()
    # Patch both ModelStore and bulk_insert_alerts
    with patch("api.routers.predict.ModelStore") as mock_store, \
         patch("api.routers.predict.alerts_dao.bulk_insert_alerts") as mock_insert:
        mock_store.get.return_value.get_active.return_value = bundle

        from db.models import Alert
        from datetime import datetime, timezone
        import json

        fake_alert = Alert(
            id=1, flow_id="0", consensus="KNOWN_ATTACK", severity="high",
            predicted_label=1, classifier_confidence=0.9, ae_score=0.1,
            ae_threshold=0.5, top_features_json=json.dumps([]),
            analyst_verdict=None, feedback_at=None,
            timestamp=datetime.now(timezone.utc),
        )
        mock_insert.return_value = [fake_alert]

        content = _make_csv_bytes(FEATURE_NAMES, n_rows=3)
        resp = await client.post(
            "/api/predict/batch",
            files={"file": ("flows.csv", content, "text/csv")},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert "alerts" in data
    assert "summary" in data
    assert data["summary"]["total"] >= 0


async def test_predict_job_status_404(client: AsyncClient):
    resp = await client.get("/api/predict/nonexistent-job/status")
    assert resp.status_code == 404


async def test_predict_job_results_404(client: AsyncClient):
    resp = await client.get("/api/predict/nonexistent-job/results")
    assert resp.status_code == 404


async def test_predict_job_results_409_not_done(client: AsyncClient):
    from api.http import jobs as _jobs
    import threading
    barrier = threading.Event()

    def _slow(*args):
        barrier.wait(timeout=5)
        return {}

    job_id = "test-async-job-id"
    _jobs.launch_predict_job(job_id, _slow)

    resp = await client.get(f"/api/predict/{job_id}/results")
    assert resp.status_code == 409
    barrier.set()

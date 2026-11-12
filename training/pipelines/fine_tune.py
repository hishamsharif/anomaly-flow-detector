from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable

import numpy as np

from api.db.crud import sync_set_active, sync_update_model_version
from core.registry import ANOMALY_DETECTOR_REGISTRY
from core.store import ModelStore
from training.data_loader import load_dataset
from training.feature_eng.pipeline import FeatureEngResult


def _apply_saved_transforms(df, bundle) -> tuple[np.ndarray, np.ndarray]:
    """Apply saved masks + detector_scaler to raw DataFrame. Returns (X_scaled, y)."""
    feature_cols = [c for c in df.columns if c != "label"]
    X_raw = df[feature_cols].values.astype(np.float32)
    y = df["label"].values

    if X_raw.shape[1] != bundle.variance_mask.shape[0]:
        raise ValueError(
            f"Feature count mismatch: data has {X_raw.shape[1]} columns, "
            f"bundle variance_mask expects {bundle.variance_mask.shape[0]}"
        )

    X_v = X_raw[:, bundle.variance_mask]
    X_vc = X_v[:, bundle.corr_mask]
    X_vcs = X_vc[:, bundle.selector_mask]
    return bundle.detector_scaler.transform(X_vcs), y


def run_ae_fine_tune(
    run_id: str,
    version_id: int,
    version_name: str,
    classifier_type: str,
    detector_type: str,
    prev_artifact_path: str,
    settings,
    emit: Callable[[str, str], None],
    db_url: str,
    analyst_flows: list[dict] | None = None,
) -> None:
    training_start_time = time.time()
    try:
        # 1. Load previous bundle so we have its masks + scalers
        emit("progress", "Loading previous model bundle …")
        store = ModelStore.get()
        bundle = store.load_into_memory(prev_artifact_path)

        # 2. Resolve device
        device = settings.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # 3. Load fresh AE instance with correct device, then restore saved weights
        detector = ANOMALY_DETECTOR_REGISTRY[detector_type](device=device)
        detector.load(prev_artifact_path)
        emit("step", "10")

        # 4. Load dataset + apply saved transforms (no re-fitting of masks or scalers)
        emit("progress", "Loading dataset and applying saved transforms …")
        train_df, val_df, test_df = load_dataset(settings.bundled_dataset_dir)
        X_tr_scaled, y_tr_raw = _apply_saved_transforms(train_df, bundle)
        X_va_scaled, y_va_raw = _apply_saved_transforms(val_df, bundle)
        X_te_scaled, y_te_raw = _apply_saved_transforms(test_df, bundle)
        emit("step", "25")

        # 5. Build replay buffer: historical benign rows + analyst-confirmed SAFE flows (HITL)
        emit("progress", "Building replay buffer (historical benign + analyst-confirmed flows) …")
        benign_tr_mask = y_tr_raw == 0
        X_tr_benign = X_tr_scaled[benign_tr_mask]
        rng = np.random.default_rng(42)
        n_replay = max(1, int(len(X_tr_benign) * settings.ae_replay_buffer_fraction))
        idx = rng.choice(len(X_tr_benign), size=min(n_replay, len(X_tr_benign)), replace=False)
        X_replay = X_tr_benign[idx]

        # Extend with analyst-confirmed benign flows passed in from the router (HITL differentiator).
        # features_json stores post-filter feature dicts keyed by bundle.feature_names.
        if analyst_flows:
            try:
                X_analyst_raw = np.array(
                    [[flow.get(name, 0.0) for name in bundle.feature_names] for flow in analyst_flows],
                    dtype=np.float32,
                )
                X_analyst_scaled = bundle.detector_scaler.transform(X_analyst_raw)
                X_replay = np.vstack([X_replay, X_analyst_scaled])
                emit("progress", f"  HITL: added {len(X_analyst_scaled)} analyst-confirmed benign flows to replay buffer.")
            except Exception as exc:
                # Analyst data is advisory — never block fine-tune on it
                emit("progress", f"  HITL: skipped analyst flows ({exc})")
        emit("step", "35")

        # 6. Fine-tune: encoder frozen for ae_freeze_epochs, then fully unfrozen
        emit("progress", f"Fine-tuning AE ({settings.ae_freeze_epochs} frozen-encoder epochs) …")
        detector.fine_tune(X_replay, freeze_epochs=settings.ae_freeze_epochs)
        emit("step", "65")

        # 7. Recalibrate threshold on val-benign
        emit("progress", "Recalibrating AE threshold on val-benign …")
        benign_va_mask = y_va_raw == 0
        X_va_benign = X_va_scaled[benign_va_mask] if benign_va_mask.any() else X_va_scaled
        detector.calibrate_threshold(X_va_benign, percentile=settings.ae_anomaly_threshold_percentile)
        novel_critical_threshold = float(np.percentile(detector.score(X_va_benign), 75.0))
        emit("step", "75")

        # 8. Regression gate (advisory) — AE-based FAR and AMR on test set
        emit("progress", "Computing regression gate metrics …")
        benign_te_mask = y_te_raw == 0
        attack_te_mask = y_te_raw == 1
        scores_te_benign = detector.score(X_te_scaled[benign_te_mask]) if benign_te_mask.any() else np.array([])
        scores_te_attack = detector.score(X_te_scaled[attack_te_mask]) if attack_te_mask.any() else np.array([])
        ae_far = float((scores_te_benign > detector.threshold).mean()) if len(scores_te_benign) else 0.0
        ae_amr = float((scores_te_attack <= detector.threshold).mean()) if len(scores_te_attack) else 0.0

        old_metrics: dict = {}
        prev_meta = Path(prev_artifact_path) / "metadata.json"
        if prev_meta.exists():
            old_metrics = json.loads(prev_meta.read_text())

        old_ae_far = old_metrics.get("ae_far")
        old_ae_amr = old_metrics.get("ae_amr")
        if old_ae_far is not None and (ae_far - old_ae_far) > settings.far_gate_max:
            emit("progress", f"⚠ Regression gate: FAR {ae_far:.4f} vs {old_ae_far:.4f} (+{ae_far - old_ae_far:.4f}, gate {settings.far_gate_max})")
        if old_ae_amr is not None and (ae_amr - old_ae_amr) > settings.recall_gate_max:
            emit("progress", f"⚠ Regression gate: AMR {ae_amr:.4f} vs {old_ae_amr:.4f} (+{ae_amr - old_ae_amr:.4f}, gate {settings.recall_gate_max})")
        emit("step", "80")

        # 9. Assemble metrics
        metrics = {
            "classifier_type": classifier_type,
            "detector_type": detector_type,
            "fine_tune_run": True,
            "fine_tune_from": prev_artifact_path,
            "ae_threshold": round(detector.threshold, 6),
            "novel_critical_threshold": round(novel_critical_threshold, 6),
            "ae_far": round(ae_far, 6),
            "ae_amr": round(ae_amr, 6),
            "replay_buffer_size": int(len(X_replay)),
            "n_features_final": len(bundle.feature_names),
            "training_duration_s": round(time.time() - training_start_time, 1),
            # carry classifier metrics forward for reference in TrainingPage
            "f1": old_metrics.get("f1"),
            "pr_auc": old_metrics.get("pr_auc"),
            "known_attack_f1": old_metrics.get("known_attack_f1"),
            "far": old_metrics.get("far"),
            "amr": old_metrics.get("amr"),
        }

        # 10. Build feat_result from bundle (preserves saved masks + scalers)
        feat_result = FeatureEngResult(
            variance_mask=bundle.variance_mask,
            corr_mask=bundle.corr_mask,
            selector_mask=bundle.selector_mask,
            feature_names=bundle.feature_names,
            classifier_scaler=bundle.classifier_scaler,
            detector_scaler=bundle.detector_scaler,
        )

        # 11. Save artifacts atomically — classifier.joblib copied from prev (I5)
        emit("progress", "Saving artifacts …")
        artifact_path = store.save_artifacts(
            run_id=run_id,
            name=version_name.replace(" ", "_").replace("/", "-").replace("—", "-"),
            clf=None,
            ae=detector,
            feat_result=feat_result,
            metrics=metrics,
            artifact_dir=settings.model_artifact_dir,
            copy_clf_from=prev_artifact_path,
        )
        sync_update_model_version(db_url, version_id, path=artifact_path, metrics_json=json.dumps(metrics))
        emit("step", "95")

        # 12. Auto-promote
        sync_set_active(db_url, version_id)
        store.load_into_memory(artifact_path)
        emit("step", "100")
        emit("done", f"AE fine-tune complete — {version_name} active")

    except Exception as exc:
        sync_update_model_version(db_url, version_id, status="failed")
        emit("error", f"{type(exc).__name__}: {exc}")

from __future__ import annotations

import json
import time
from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    f1_score,
)

from api.db.crud import sync_set_active, sync_update_model_version
from core.registry import ANOMALY_DETECTOR_REGISTRY, CLASSIFIER_REGISTRY
from core.store import ModelStore
from training.data_loader import load_dataset
from training.feature_eng.pipeline import run_feature_engineering


def run_initial_training(
    run_id: str,
    version_id: int,
    version_name: str,
    classifier_type: str,
    detector_type: str,
    settings,
    emit: Callable[[str, str], None],
    db_url: str,
) -> None:
    training_start_time = time.time()
    try:
        # 1. Load data
        emit("progress", "Loading CIC-IDS2017 …")
        train_df, val_df, test_df = load_dataset(settings.bundled_dataset_dir)
        emit("step", "10")

        # 2. Feature engineering
        emit("progress", "Running feature engineering …")
        (
            X_tr_classifier, X_tr_detector,
            X_va_classifier, X_va_detector,
            X_te_classifier, X_te_detector,
            y_tr, y_te,
            y_va,
            feat_result,
        ) = run_feature_engineering(train_df, val_df, test_df, settings)
        emit("step", "30")

        # Resolve device
        device = settings.device
        if device == "auto":
            try:
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                device = "cpu"

        # 3. Fit classifier
        emit("progress", f"Training {CLASSIFIER_REGISTRY[classifier_type].label} …")
        classifier_class = CLASSIFIER_REGISTRY[classifier_type]
        classifier = classifier_class(device=device)
        classifier.fit(X_tr_classifier, y_tr)
        emit("step", "60")

        # 4. Fit AE on benign-only train, calibrate on val-benign
        emit("progress", "Training MLP Autoencoder on benign flows …")
        detector_class = ANOMALY_DETECTOR_REGISTRY[detector_type]
        detector = detector_class(device=device)
        detector.fit(X_tr_detector)
        benign_val_mask = y_va == 0
        X_va_benign = X_va_detector[benign_val_mask] if benign_val_mask.any() else X_va_detector
        detector.calibrate_threshold(X_va_benign, percentile=settings.ae_anomaly_threshold_percentile)
        # p75 of val-benign errors → severity threshold for NOVEL_ATTACK (critical vs high, §3.1)
        novel_critical_threshold = float(np.percentile(detector.score(X_va_benign), 75.0))
        emit("step", "80")

        # 5. Evaluate on test set
        emit("progress", "Evaluating …")
        test_labels, test_confidences = classifier.predict(X_te_classifier)
        test_ae_scores = detector.score(X_te_detector)

        f1 = float(f1_score(y_te, test_labels, average="weighted", zero_division=0))
        pr_auc = float(average_precision_score(y_te, test_confidences) if len(np.unique(y_te)) > 1 else 0.0)

        # FAR = FP / (FP + TN)  on BENIGN rows
        benign_te = y_te == 0
        if benign_te.any():
            fp = int(((test_labels == 1) & benign_te).sum())
            tn = int(((test_labels == 0) & benign_te).sum())
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            far = 0.0

        # AMR = FN / (FN + TP) on attack rows
        attack_te = y_te == 1
        if attack_te.any():
            fn = int(((test_labels == 0) & attack_te).sum())
            tp = int(((test_labels == 1) & attack_te).sum())
            amr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        else:
            amr = 0.0

        known_attack_f1 = float(f1_score(y_te, test_labels, pos_label=1, average="binary", zero_division=0))

        metrics = {
            "classifier_type": classifier_type,
            "detector_type": detector_type,
            "f1": round(f1, 6),
            "far": round(far, 6),
            "amr": round(amr, 6),
            "pr_auc": round(pr_auc, 6),
            "known_attack_f1": round(known_attack_f1, 6),
            "n_features_final": len(feat_result.feature_names),
            "ae_threshold": round(detector.threshold, 6),
            "novel_critical_threshold": round(novel_critical_threshold, 6),
            "training_duration_s": round(time.time() - training_start_time, 1),
            "train_rows": len(y_tr),
            "val_rows": len(y_va),
            "test_rows": len(y_te),
        }
        emit("step", "90")

        # 6. Save artifacts atomically
        emit("progress", "Saving artifacts …")
        store = ModelStore.get()
        artifact_path = store.save_artifacts(
            run_id=run_id,
            name=version_name.replace(" ", "_").replace("/", "-").replace("—", "-"),
            clf=classifier,
            ae=detector,
            feat_result=feat_result,
            metrics=metrics,
            artifact_dir=settings.model_artifact_dir,
        )
        sync_update_model_version(db_url, version_id, path=artifact_path, metrics_json=json.dumps(metrics))
        emit("step", "95")

        # 7. Auto-promote to active
        sync_set_active(db_url, version_id)
        store.load_into_memory(artifact_path)
        emit("step", "100")
        emit("done", f"Training complete — {version_name} active")

    except Exception as exc:
        sync_update_model_version(db_url, version_id, status="failed")
        emit("error", f"{type(exc).__name__}: {exc}")

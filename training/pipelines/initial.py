from __future__ import annotations

import time
import traceback
from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
)

from api.db.crud import sync_set_active, sync_update_model_version
from core.registry import CLASSIFIERS, DETECTORS
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
    start = time.time()
    try:
        # 1. Load data
        emit("progress", "Loading CIC-IDS2017 …")
        train_df, val_df, test_df = load_dataset(settings.bundled_dataset_dir)
        emit("step", "10")

        # 2. Feature engineering
        emit("progress", "Running feature engineering …")
        (
            X_tr_clf, X_tr_det,
            X_va_clf, X_va_det,
            X_te_clf, X_te_det,
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
        emit("progress", f"Training {CLASSIFIERS[classifier_type].label} …")
        clf_cls = CLASSIFIERS[classifier_type]
        clf = clf_cls(device=device)
        clf.fit(X_tr_clf, y_tr)
        emit("step", "60")

        # 4. Fit AE on benign-only train, calibrate on val-benign
        emit("progress", "Training MLP Autoencoder on benign flows …")
        ae_cls = DETECTORS[detector_type]
        ae = ae_cls(device=device)
        ae.fit(X_tr_det)
        benign_val_mask = y_va == 0
        X_va_benign = X_va_det[benign_val_mask] if benign_val_mask.any() else X_va_det
        ae.calibrate_threshold(X_va_benign, percentile=settings.ae_anomaly_threshold_percentile)
        emit("step", "80")

        # 5. Evaluate on test set
        emit("progress", "Evaluating …")
        labels_te, confs_te = clf.predict(X_te_clf)
        ae_scores_te = ae.score(X_te_det)

        f1 = float(f1_score(y_te, labels_te, average="weighted", zero_division=0))
        pr_auc = float(average_precision_score(y_te, confs_te) if len(np.unique(y_te)) > 1 else 0.0)

        # FAR = FP / (FP + TN)  on BENIGN rows
        benign_te = y_te == 0
        if benign_te.any():
            fp = int(((labels_te == 1) & benign_te).sum())
            tn = int(((labels_te == 0) & benign_te).sum())
            far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        else:
            far = 0.0

        # AMR = FN / (FN + TP) on attack rows
        attack_te = y_te == 1
        if attack_te.any():
            fn = int(((labels_te == 0) & attack_te).sum())
            tp = int(((labels_te == 1) & attack_te).sum())
            amr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        else:
            amr = 0.0

        known_attack_f1 = float(f1_score(y_te, labels_te, pos_label=1, average="binary", zero_division=0))

        metrics = {
            "f1": round(f1, 6),
            "far": round(far, 6),
            "amr": round(amr, 6),
            "pr_auc": round(pr_auc, 6),
            "known_attack_f1": round(known_attack_f1, 6),
            "n_features_final": len(feat_result.feature_names),
            "ae_threshold": round(ae.threshold, 6),
            "training_duration_s": round(time.time() - start, 1),
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
            clf=clf,
            ae=ae,
            feat_result=feat_result,
            metrics=metrics,
            artifact_dir=settings.model_artifact_dir,
        )
        sync_update_model_version(db_url, version_id, path=artifact_path, metrics_json=__import__("json").dumps(metrics))
        emit("step", "95")

        # 7. Auto-promote to active
        sync_set_active(db_url, version_id)
        store.load_into_memory(artifact_path)
        emit("step", "100")
        emit("done", f"Training complete — {version_name} active")

    except Exception as exc:
        sync_update_model_version(db_url, version_id, status="failed")
        emit("error", f"{type(exc).__name__}: {exc}")

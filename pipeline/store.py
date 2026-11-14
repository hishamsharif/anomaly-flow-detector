from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np


@dataclass
class ArtifactBundle:
    clf: Any
    ae: Any
    classifier_scaler: Any
    detector_scaler: Any
    feature_names: list[str]
    variance_mask: np.ndarray
    corr_mask: np.ndarray
    selector_mask: np.ndarray
    ae_threshold: float
    novel_critical_threshold: float
    ae_outlier_clip_sigma: float = 5.0


class ModelStore:
    _instance: ModelStore | None = None
    _active: ArtifactBundle | None = None

    @classmethod
    def get(cls) -> ModelStore:
        if cls._instance is None:
            cls._instance = ModelStore()
        return cls._instance

    def save_artifacts(
        self,
        run_id: str,
        name: str,
        clf: Any,
        ae: Any,
        feat_result: Any,  # FeatureEngResult
        metrics: dict,
        artifact_dir: str,
        copy_clf_from: str | None = None,
    ) -> str:
        base = Path(artifact_dir)
        tmp = base / f"tmp_{run_id}"
        final = base / name
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            if copy_clf_from is not None:
                # I5: copy classifier bytes unchanged so XGBoost artifact is never re-serialised
                shutil.copy2(Path(copy_clf_from) / "classifier.joblib", tmp / "classifier.joblib")
            else:
                clf.save(str(tmp))
            ae.save(str(tmp))
            joblib.dump(feat_result.classifier_scaler, tmp / "clf_scaler.joblib")
            joblib.dump(feat_result.detector_scaler, tmp / "det_scaler.joblib")
            np.save(str(tmp / "variance_mask.npy"), feat_result.variance_mask)
            np.save(str(tmp / "corr_mask.npy"), feat_result.corr_mask)
            np.save(str(tmp / "selector_mask.npy"), feat_result.selector_mask)
            (tmp / "feature_names.json").write_text(json.dumps(feat_result.feature_names))
            (tmp / "metadata.json").write_text(json.dumps(metrics, default=str))
            shutil.move(str(tmp), str(final))
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return str(final)

    def load_into_memory(self, path: str) -> ArtifactBundle:
        from pipeline.models.registry import ANOMALY_DETECTOR_REGISTRY, CLASSIFIER_REGISTRY

        p = Path(path)
        meta = json.loads((p / "metadata.json").read_text())

        clf_type = meta.get("classifier_type", "xgboost")
        det_type = meta.get("detector_type", "mlp_ae")
        clf = CLASSIFIER_REGISTRY[clf_type]()
        clf.load(str(p))
        ae = ANOMALY_DETECTOR_REGISTRY[det_type]()
        ae.load(str(p))

        classifier_scaler = joblib.load(p / "clf_scaler.joblib")
        detector_scaler = joblib.load(p / "det_scaler.joblib")
        feature_names = json.loads((p / "feature_names.json").read_text())
        variance_mask = np.load(str(p / "variance_mask.npy"))
        corr_mask = np.load(str(p / "corr_mask.npy"))
        selector_mask = np.load(str(p / "selector_mask.npy"))
        ae_threshold = float(meta.get("ae_threshold", ae.threshold))
        novel_critical_threshold = float(meta.get("novel_critical_threshold", ae_threshold * 0.95))
        ae_outlier_clip_sigma = float(meta.get("ae_outlier_clip_sigma", 5.0))

        bundle = ArtifactBundle(
            clf=clf,
            ae=ae,
            classifier_scaler=classifier_scaler,
            detector_scaler=detector_scaler,
            feature_names=feature_names,
            variance_mask=variance_mask,
            corr_mask=corr_mask,
            selector_mask=selector_mask,
            ae_threshold=ae_threshold,
            novel_critical_threshold=novel_critical_threshold,
            ae_outlier_clip_sigma=ae_outlier_clip_sigma,
        )
        self.__class__._active = bundle
        return bundle

    def get_active(self) -> ArtifactBundle | None:
        return self.__class__._active

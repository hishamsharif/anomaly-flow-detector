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
    clf_scaler: Any
    det_scaler: Any
    feature_names: list[str]
    variance_mask: np.ndarray
    corr_mask: np.ndarray
    ae_threshold: float


class ModelStore:
    _instance: "ModelStore | None" = None
    _active: ArtifactBundle | None = None

    @classmethod
    def get(cls) -> "ModelStore":
        if cls._instance is None:
            cls._instance = ModelStore()
        return cls._instance

    def save_artifacts(
        self,
        run_id: str,
        name: str,
        clf: Any,
        ae: Any,
        feat_result: Any,         # FeatureEngResult
        metrics: dict,
        artifact_dir: str,
    ) -> str:
        base = Path(artifact_dir)
        tmp = base / f"tmp_{run_id}"
        final = base / name
        tmp.mkdir(parents=True, exist_ok=True)
        try:
            clf.save(str(tmp))
            ae.save(str(tmp))
            joblib.dump(feat_result.clf_scaler, tmp / "clf_scaler.joblib")
            joblib.dump(feat_result.det_scaler, tmp / "det_scaler.joblib")
            np.save(str(tmp / "variance_mask.npy"), feat_result.variance_mask)
            np.save(str(tmp / "corr_mask.npy"), feat_result.corr_mask)
            (tmp / "feature_names.json").write_text(json.dumps(feat_result.feature_names))
            (tmp / "metadata.json").write_text(json.dumps(metrics, default=str))
            shutil.move(str(tmp), str(final))
        except Exception:
            shutil.rmtree(tmp, ignore_errors=True)
            raise
        return str(final)

    def load_into_memory(self, path: str) -> ArtifactBundle:
        from core.registry import CLASSIFIERS, DETECTORS

        p = Path(path)
        meta = json.loads((p / "metadata.json").read_text())

        clf_type = meta.get("classifier_type", "xgboost")
        det_type = meta.get("detector_type", "mlp_ae")
        clf = CLASSIFIERS[clf_type]()
        clf.load(str(p))
        ae = DETECTORS[det_type]()
        ae.load(str(p))

        clf_scaler = joblib.load(p / "clf_scaler.joblib")
        det_scaler = joblib.load(p / "det_scaler.joblib")
        feature_names = json.loads((p / "feature_names.json").read_text())
        variance_mask = np.load(str(p / "variance_mask.npy"))
        corr_mask = np.load(str(p / "corr_mask.npy"))
        ae_threshold = float(meta.get("ae_threshold", ae.threshold))

        bundle = ArtifactBundle(
            clf=clf,
            ae=ae,
            clf_scaler=clf_scaler,
            det_scaler=det_scaler,
            feature_names=feature_names,
            variance_mask=variance_mask,
            corr_mask=corr_mask,
            ae_threshold=ae_threshold,
        )
        self.__class__._active = bundle
        return bundle

    def get_active(self) -> ArtifactBundle | None:
        return self.__class__._active

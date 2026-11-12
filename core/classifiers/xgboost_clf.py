from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import xgboost as xgb

from core.base import PreprocessingSpec


class XGBoostClassifier:
    label = "XGBoost Classifier"
    preprocessing = PreprocessingSpec(needs_scaling=True, preferred_scalers=("standard",))

    def __init__(self, device: str = "cpu") -> None:
        self._device = device
        self._xgb_model: xgb.XGBClassifier | None = None

    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> None:
        self._xgb_model = xgb.XGBClassifier(
            device=self._device,
            eval_metric="logloss",
            random_state=42,
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
        )
        self._xgb_model.fit(X, y)

    @property
    def feature_importances(self) -> np.ndarray | None:
        return self._xgb_model.feature_importances_ if self._xgb_model is not None else None

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        assert self._xgb_model is not None, "fit() first"
        labels = self._xgb_model.predict(X)
        class_proba = self._xgb_model.predict_proba(X)
        confidences = class_proba[:, 1]   # P(attack) — correct signal for PR-AUC and confidence threshold
        return labels, confidences

    def save(self, path: str) -> None:
        assert self._xgb_model is not None, "fit() first"
        Path(path).mkdir(parents=True, exist_ok=True)
        joblib.dump(self._xgb_model, Path(path) / "classifier.joblib")

    def load(self, path: str) -> None:
        self._xgb_model = joblib.load(Path(path) / "classifier.joblib")

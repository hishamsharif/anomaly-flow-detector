from __future__ import annotations

from typing import Any

from core.classifiers.xgboost_clf import XGBoostClassifier
from core.detectors.mlp_ae import MLPAutoencoder

# Human-readable labels — only include types that are registered and implemented.
# UI dropdowns are populated from these; unregistered entries cause 422 on training start.
CLASSIFIER_LABELS: dict[str, str] = {
    "xgboost": "XGBoost Classifier",
}

DETECTOR_LABELS: dict[str, str] = {
    "mlp_ae": "MLP Autoencoder",
}

CLASSIFIER_REGISTRY: dict[str, Any] = {
    "xgboost": XGBoostClassifier,
}

ANOMALY_DETECTOR_REGISTRY: dict[str, Any] = {
    "mlp_ae": MLPAutoencoder,
}

from __future__ import annotations

from typing import Any

from core.classifiers.xgboost_clf import XGBoostClassifier
from core.detectors.mlp_ae import MLPAutoencoder

# Human-readable labels served by GET /api/models/available-types
CLASSIFIER_LABELS: dict[str, str] = {
    "xgboost": "XGBoost Classifier",
    "random_forest": "Random Forest",
}

DETECTOR_LABELS: dict[str, str] = {
    "mlp_ae": "MLP Autoencoder",
    "isolation_forest": "Isolation Forest",
}

CLASSIFIERS: dict[str, Any] = {
    "xgboost": XGBoostClassifier,
}

DETECTORS: dict[str, Any] = {
    "mlp_ae": MLPAutoencoder,
}

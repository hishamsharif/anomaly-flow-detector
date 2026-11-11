from __future__ import annotations

from typing import Any

# Human-readable labels served by GET /api/models/available-types
CLASSIFIER_LABELS: dict[str, str] = {
    "xgboost": "XGBoost Classifier",
    "random_forest": "Random Forest",
}

DETECTOR_LABELS: dict[str, str] = {
    "mlp_ae": "MLP Autoencoder",
    "isolation_forest": "Isolation Forest",
}

# Class implementations populated in Slice 2
CLASSIFIERS: dict[str, Any] = {}
DETECTORS: dict[str, Any] = {}

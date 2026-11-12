from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from training.feature_eng.correlation_filter import CorrelationFilter
from training.feature_eng.selector import FeatureSelector
from training.feature_eng.variance_filter import VarianceFilter


@dataclass
class FeatureEngResult:
    variance_mask: np.ndarray
    corr_mask: np.ndarray
    selector_mask: np.ndarray
    feature_names: list[str]
    classifier_scaler: StandardScaler
    detector_scaler: StandardScaler


def run_feature_engineering(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    settings,
) -> tuple[
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray, np.ndarray,
    np.ndarray,
    FeatureEngResult,
]:
    feature_cols = [c for c in train_df.columns if c != "label"]

    X_tr = train_df[feature_cols].values.astype(np.float32)
    y_tr = train_df["label"].values
    X_va = val_df[feature_cols].values.astype(np.float32)
    y_va = val_df["label"].values
    X_te = test_df[feature_cols].values.astype(np.float32)
    y_te = test_df["label"].values

    # Step 1 — variance filter
    variance_filter = VarianceFilter(threshold=settings.variance_threshold)
    X_tr = variance_filter.fit_transform(X_tr)
    X_va = variance_filter.transform(X_va)
    X_te = variance_filter.transform(X_te)
    names_after_vf = [n for n, keep in zip(feature_cols, variance_filter.mask) if keep]

    # Step 2 — correlation filter
    correlation_filter = CorrelationFilter(threshold=settings.correlation_threshold)
    X_tr = correlation_filter.fit_transform(X_tr, names_after_vf)
    X_va = correlation_filter.transform(X_va)
    X_te = correlation_filter.transform(X_te)
    names_after_cf = [n for n, keep in zip(names_after_vf, correlation_filter.mask) if keep]

    # Step 3 — SMOTE on train only
    if len(np.unique(y_tr)) > 1:
        try:
            from imblearn.over_sampling import SMOTE
            counts = np.bincount(y_tr.astype(int))
            sampling = {cls: min(cnt, settings.smote_max_per_class) for cls, cnt in enumerate(counts) if cnt > 0}
            sm = SMOTE(sampling_strategy=sampling, random_state=42)
            X_tr, y_tr = sm.fit_resample(X_tr, y_tr)
        except Exception:
            pass  # SMOTE fails on very small datasets; proceed without

    # Step 4 — optional feature selection
    feature_selector = FeatureSelector(k=settings.feature_selection_k, method=settings.feature_selection_method)
    X_tr = feature_selector.fit_transform(X_tr, y_tr, names_after_cf)
    X_va = feature_selector.transform(X_va)
    X_te = feature_selector.transform(X_te)
    final_names = feature_selector.selected_names

    # Step 5a — classifier scaler (fit on train)
    classifier_scaler = StandardScaler()
    X_tr_classifier = classifier_scaler.fit_transform(X_tr)
    X_va_classifier = classifier_scaler.transform(X_va)
    X_te_classifier = classifier_scaler.transform(X_te)

    # Step 5b — detector scaler (fit on benign-only train)
    benign_mask = y_tr == 0
    X_tr_benign = X_tr[benign_mask]
    detector_scaler = StandardScaler()
    if len(X_tr_benign) == 0:
        X_tr_benign = X_tr
    detector_scaler.fit(X_tr_benign)
    X_tr_detector = detector_scaler.transform(X_tr_benign)
    X_va_detector = detector_scaler.transform(X_va)
    X_te_detector = detector_scaler.transform(X_te)

    result = FeatureEngResult(
        variance_mask=variance_filter.mask,
        corr_mask=correlation_filter.mask,
        selector_mask=feature_selector.mask,
        feature_names=final_names,
        classifier_scaler=classifier_scaler,
        detector_scaler=detector_scaler,
    )

    return (
        X_tr_classifier, X_tr_detector,
        X_va_classifier, X_va_detector,
        X_te_classifier, X_te_detector,
        y_tr, y_te,
        y_va,
        result,
    )

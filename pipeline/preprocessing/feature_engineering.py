from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from pipeline.preprocessing.correlation_filter import CorrelationFilter
from pipeline.preprocessing.selector import FeatureSelector
from pipeline.preprocessing.variance_filter import VarianceFilter


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
    np.ndarray,
    np.ndarray,  # X_train_classifier, X_train_detector
    np.ndarray,
    np.ndarray,  # X_val_classifier,   X_val_detector
    np.ndarray,
    np.ndarray,  # X_test_classifier,  X_test_detector
    np.ndarray,
    np.ndarray,
    np.ndarray,  # y_train, y_val, y_test
    FeatureEngResult,
]:
    """Apply feature engineering steps and return scaled matrices for each split.

    X_*_clf  — classifier-scaler (StandardScaler fit on full train)
    X_*_det  — detector-scaler  (StandardScaler fit on benign-only train)
    X_train_det contains only benign rows (AE trains on benign-only data).
    """
    # Intersect columns across all three splits to handle schema differences
    # between dataset files (e.g. one stub file missing a column).
    shared_cols = (set(train_df.columns) & set(val_df.columns) & set(test_df.columns)) - {"label"}
    feature_cols = [c for c in train_df.columns if c in shared_cols]

    X_train = train_df[feature_cols].values.astype(np.float32)
    y_train = train_df["label"].values
    X_val = val_df[feature_cols].values.astype(np.float32)
    y_val = val_df["label"].values
    X_test = test_df[feature_cols].values.astype(np.float32)
    y_test = test_df["label"].values

    # Step 1 — variance filter (fit on train, applied to all splits)
    variance_filter = VarianceFilter(threshold=settings.variance_threshold)
    X_train = variance_filter.fit_transform(X_train)
    X_val = variance_filter.transform(X_val)
    X_test = variance_filter.transform(X_test)
    names_after_variance = [n for n, keep in zip(feature_cols, variance_filter.mask) if keep]

    # Step 2 — correlation filter (fit on train, applied to all splits)
    correlation_filter = CorrelationFilter(threshold=settings.correlation_threshold)
    X_train = correlation_filter.fit_transform(X_train, names_after_variance)
    X_val = correlation_filter.transform(X_val)
    X_test = correlation_filter.transform(X_test)
    names_after_correlation = [
        n for n, keep in zip(names_after_variance, correlation_filter.mask) if keep
    ]

    # Step 3 — SMOTE on train only (balances attack classes;
    # val/test untouched for realistic evaluation)
    if len(np.unique(y_train)) > 1:
        try:
            from imblearn.over_sampling import SMOTE

            counts = np.bincount(y_train.astype(int))
            sampling = {
                cls: min(cnt, settings.smote_max_per_class)
                for cls, cnt in enumerate(counts)
                if cnt > 0
            }
            sm = SMOTE(sampling_strategy=sampling, random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        except Exception:
            pass  # SMOTE fails on very small datasets; proceed without

    # Step 4 — optional supervised feature selection (fit on train)
    feature_selector = FeatureSelector(
        k=settings.feature_selection_k, method=settings.feature_selection_method
    )
    X_train = feature_selector.fit_transform(X_train, y_train, names_after_correlation)
    X_val = feature_selector.transform(X_val)
    X_test = feature_selector.transform(X_test)
    final_names = feature_selector.selected_names

    # Step 5a — classifier scaler: StandardScaler fit on full train (all classes)
    classifier_scaler = StandardScaler()
    X_train_classifier = classifier_scaler.fit_transform(X_train)
    X_val_classifier = classifier_scaler.transform(X_val)
    X_test_classifier = classifier_scaler.transform(X_test)

    # Step 5b — detector scaler: StandardScaler fit on benign-only train
    # AE learns the distribution of normal traffic; fitting on attack rows would bias the scaler.
    benign_train_mask = y_train == 0
    X_train_benign = X_train[benign_train_mask] if benign_train_mask.any() else X_train
    detector_scaler = StandardScaler()
    detector_scaler.fit(X_train_benign)
    _clip = getattr(settings, "ae_outlier_clip_sigma", 5.0)
    X_train_detector = np.clip(detector_scaler.transform(X_train_benign), -_clip, _clip)
    X_val_detector = np.clip(detector_scaler.transform(X_val), -_clip, _clip)
    X_test_detector = np.clip(detector_scaler.transform(X_test), -_clip, _clip)

    result = FeatureEngResult(
        variance_mask=variance_filter.mask,
        corr_mask=correlation_filter.mask,
        selector_mask=feature_selector.mask,
        feature_names=final_names,
        classifier_scaler=classifier_scaler,
        detector_scaler=detector_scaler,
    )

    return (
        X_train_classifier,
        X_train_detector,
        X_val_classifier,
        X_val_detector,
        X_test_classifier,
        X_test_detector,
        y_train,
        y_val,
        y_test,
        result,
    )

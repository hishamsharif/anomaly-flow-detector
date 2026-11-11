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
    clf_scaler: StandardScaler
    det_scaler: StandardScaler


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
    vf = VarianceFilter(threshold=settings.variance_threshold)
    X_tr = vf.fit_transform(X_tr)
    X_va = vf.transform(X_va)
    X_te = vf.transform(X_te)
    names_after_vf = [n for n, keep in zip(feature_cols, vf.mask) if keep]

    # Step 2 — correlation filter
    cf = CorrelationFilter(threshold=settings.correlation_threshold)
    X_tr = cf.fit_transform(X_tr, names_after_vf)
    X_va = cf.transform(X_va)
    X_te = cf.transform(X_te)
    names_after_cf = [n for n, keep in zip(names_after_vf, cf.mask) if keep]

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
    sel = FeatureSelector(k=settings.feature_selection_k, method=settings.feature_selection_method)
    X_tr = sel.fit_transform(X_tr, y_tr, names_after_cf)
    X_va = sel.transform(X_va)
    X_te = sel.transform(X_te)
    final_names = sel.selected_names

    # Step 5a — clf scaler (fit on train)
    clf_scaler = StandardScaler()
    X_tr_clf = clf_scaler.fit_transform(X_tr)
    X_va_clf = clf_scaler.transform(X_va)
    X_te_clf = clf_scaler.transform(X_te)

    # Step 5b — det scaler (fit on benign-only train)
    benign_mask = y_tr == 0
    X_tr_benign = X_tr[benign_mask]
    det_scaler = StandardScaler()
    if len(X_tr_benign) == 0:
        X_tr_benign = X_tr
    det_scaler.fit(X_tr_benign)
    X_tr_det = det_scaler.transform(X_tr_benign)
    X_va_det = det_scaler.transform(X_va)
    X_te_det = det_scaler.transform(X_te)

    result = FeatureEngResult(
        variance_mask=vf.mask,
        corr_mask=cf.mask,
        selector_mask=sel.mask,
        feature_names=final_names,
        clf_scaler=clf_scaler,
        det_scaler=det_scaler,
    )

    return (
        X_tr_clf, X_tr_det,
        X_va_clf, X_va_det,
        X_te_clf, X_te_det,
        y_tr, y_te,
        y_va,
        result,
    )

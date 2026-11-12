from __future__ import annotations

import numpy as np
import pandas as pd


class CorrelationFilter:
    def __init__(self, threshold: float = 0.95) -> None:
        self._threshold = threshold
        self._mask: np.ndarray | None = None

    def fit(self, X: np.ndarray, feature_names: list[str]) -> "CorrelationFilter":
        feature_df = pd.DataFrame(X, columns=feature_names)
        corr_matrix = feature_df.corr(method="pearson").abs()
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        features_to_drop = {col for col in upper_triangle.columns if any(upper_triangle[col] > self._threshold)}
        features_to_keep = [n for n in feature_names if n not in features_to_drop]
        self._mask = np.array([n in features_to_keep for n in feature_names], dtype=bool)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._mask is not None, "fit() first"
        return X[:, self._mask]

    def fit_transform(self, X: np.ndarray, feature_names: list[str]) -> np.ndarray:
        return self.fit(X, feature_names).transform(X)

    @property
    def mask(self) -> np.ndarray:
        assert self._mask is not None, "fit() first"
        return self._mask

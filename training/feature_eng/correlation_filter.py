from __future__ import annotations

import numpy as np
import pandas as pd


class CorrelationFilter:
    def __init__(self, threshold: float = 0.95) -> None:
        self._threshold = threshold
        self._mask: np.ndarray | None = None

    def fit(self, X: np.ndarray, feature_names: list[str]) -> "CorrelationFilter":
        df = pd.DataFrame(X, columns=feature_names)
        corr = df.corr(method="pearson").abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop = {col for col in upper.columns if any(upper[col] > self._threshold)}
        keep = [n for n in feature_names if n not in drop]
        self._mask = np.array([n in keep for n in feature_names], dtype=bool)
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

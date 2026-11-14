from __future__ import annotations

import numpy as np


class FeatureSelector:
    def __init__(self, k: int = 0, method: str = "mi") -> None:
        self._n_top_features = k
        self._method = method
        self._mask: np.ndarray | None = None
        self._selected_feature_names: list[str] = []
        self._selector = None

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> FeatureSelector:
        n_features = X.shape[1]
        if self._n_top_features == 0 or self._method == "none":
            self._mask = np.ones(n_features, dtype=bool)
            self._selected_feature_names = list(feature_names) if feature_names else [str(i) for i in range(n_features)]
            return self

        n_to_select = min(self._n_top_features, n_features)
        if self._method == "mi":
            from sklearn.feature_selection import SelectKBest, mutual_info_classif
            self._selector = SelectKBest(mutual_info_classif, k=n_to_select)
            self._selector.fit(X, y)
            self._mask = self._selector.get_support()
        elif self._method == "rfe":
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.feature_selection import RFE
            self._selector = RFE(RandomForestClassifier(n_estimators=50, random_state=42), n_features_to_select=n_to_select)
            self._selector.fit(X, y)
            self._mask = self._selector.support_
        else:
            self._mask = np.ones(n_features, dtype=bool)

        if feature_names:
            self._selected_feature_names = [n for n, keep in zip(feature_names, self._mask) if keep]
        else:
            self._selected_feature_names = [str(i) for i, keep in enumerate(self._mask) if keep]
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._mask is not None, "fit() first"
        return X[:, self._mask]

    def fit_transform(self, X: np.ndarray, y: np.ndarray, feature_names: list[str] | None = None) -> np.ndarray:
        return self.fit(X, y, feature_names).transform(X)

    @property
    def mask(self) -> np.ndarray:
        assert self._mask is not None, "fit() first"
        return self._mask

    @property
    def selected_names(self) -> list[str]:
        return self._selected_feature_names

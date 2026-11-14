from __future__ import annotations

import warnings

import numpy as np
from sklearn.feature_selection import VarianceThreshold as _VT


class VarianceFilter:
    def __init__(self, threshold: float = 0.0) -> None:
        self._threshold = threshold
        self._selector: _VT | None = None
        self._mask: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> VarianceFilter:
        self._selector = _VT(threshold=self._threshold)
        # sklearn calls np.nanvar which emits RuntimeWarning on all-NaN columns;
        # those columns are correctly dropped, so the warning is benign.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            self._selector.fit(X)
        self._mask = self._selector.get_support()
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._selector is not None, "fit() first"
        return self._selector.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    @property
    def mask(self) -> np.ndarray:
        assert self._mask is not None, "fit() first"
        return self._mask

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


@dataclass
class PreprocessingSpec:
    needs_scaling: bool = True
    preferred_scalers: tuple[str, ...] = ("standard",)


@runtime_checkable
class BaseClassifier(Protocol):
    label: str
    preprocessing: PreprocessingSpec

    def fit(self, X: Any, y: Any, **kwargs: Any) -> None: ...
    def predict(self, X: Any) -> tuple[Any, Any]: ...  # (labels, confidences)
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...


@runtime_checkable
class BaseAnomalyDetector(Protocol):
    label: str
    preprocessing: PreprocessingSpec

    def fit(self, X: Any, **kwargs: Any) -> None: ...
    def score(self, X: Any) -> Any: ...  # reconstruction errors per sample
    def save(self, path: str) -> None: ...
    def load(self, path: str) -> None: ...

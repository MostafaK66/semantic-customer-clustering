"""Optional, injectable outlier detection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np
from sklearn.ensemble import IsolationForest

from semantic_customer_clustering.errors import (
    DependencyUnavailableError,
    ModelExecutionError,
)
from semantic_customer_clustering.models import BoolVector, FloatMatrix


class OutlierModel(Protocol):
    def fit_predict(self, data: FloatMatrix) -> object: ...


OutlierFactory = Callable[[float], OutlierModel]


class IsolationForestOutlierDetector:
    """Portable default outlier detector provided by the core dependencies."""

    def __init__(
        self,
        contamination: float,
        *,
        random_state: int = 123,
        model: OutlierModel | None = None,
    ) -> None:
        self._model = (
            model
            if model is not None
            else IsolationForest(
                contamination=contamination,
                random_state=random_state,
            )
        )

    def inlier_mask(self, data: FloatMatrix) -> BoolVector:
        """Convert Isolation Forest's 1/-1 labels into an inlier mask."""
        try:
            raw = self._model.fit_predict(data)
        except Exception as exc:
            raise ModelExecutionError(f"outlier detection failed: {exc}") from exc
        labels = np.asarray(raw)
        if labels.ndim != 1 or labels.shape[0] != data.shape[0]:
            raise ModelExecutionError("outlier detector returned malformed labels")
        if not np.isin(labels, (-1, 1)).all():
            raise ModelExecutionError("Isolation Forest labels must contain -1 and 1")
        return _ensure_enough_inliers(labels == 1)


class EcodOutlierDetector:
    """Adapter for PyOD ECOD without importing PyOD in the local test path."""

    def __init__(
        self,
        contamination: float,
        *,
        factory: OutlierFactory | None = None,
    ) -> None:
        self._contamination = contamination
        self._factory = factory

    def inlier_mask(self, data: FloatMatrix) -> BoolVector:
        """Return True for inliers and reject malformed model output."""
        factory = self._factory
        if factory is None:
            try:
                from pyod.models.ecod import ECOD
            except ImportError as exc:
                raise DependencyUnavailableError(
                    "outlier filtering requires: pip install "
                    "'semantic-customer-clustering[outliers]'"
                ) from exc
            factory = ECOD
        try:
            raw = factory(self._contamination).fit_predict(data)
        except Exception as exc:
            raise ModelExecutionError(f"outlier detection failed: {exc}") from exc
        labels = np.asarray(raw)
        if labels.ndim != 1 or labels.shape[0] != data.shape[0]:
            raise ModelExecutionError("outlier detector returned malformed labels")
        if not np.isin(labels, (0, 1)).all():
            raise ModelExecutionError("outlier labels must contain only 0 and 1")
        return _ensure_enough_inliers(labels == 0)


def _ensure_enough_inliers(raw: object) -> BoolVector:
    mask = np.asarray(raw, dtype=np.bool_)
    if np.count_nonzero(mask) < 3:
        raise ModelExecutionError("outlier filtering left fewer than three rows")
    return mask

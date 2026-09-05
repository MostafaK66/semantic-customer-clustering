"""Deterministic K-means evaluation and selection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, cast

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from semantic_customer_clustering.config import ClusteringConfig
from semantic_customer_clustering.errors import DataValidationError, ModelExecutionError
from semantic_customer_clustering.models import (
    ClusterResult,
    ClusterScore,
    FloatMatrix,
    IntVector,
)


class ClusterModel(Protocol):
    def fit_predict(self, data: FloatMatrix) -> object: ...


ClusterFactory = Callable[[int, ClusteringConfig], ClusterModel]
ScoreFunction = Callable[[FloatMatrix, IntVector], float]


def cluster_kmeans(
    data: object,
    config: ClusteringConfig,
    *,
    factory: ClusterFactory | None = None,
    scorer: ScoreFunction = silhouette_score,
) -> ClusterResult:
    """Score candidates, choose the best silhouette, and fit the final model."""
    matrix = validate_numeric_matrix(data)
    feasible = tuple(value for value in config.candidates if value < matrix.shape[0])
    if not feasible:
        raise DataValidationError(
            "no feasible cluster candidate is smaller than the number of input rows"
        )
    create = factory or _create_kmeans
    scores: list[ClusterScore] = []
    for clusters in feasible:
        labels = _labels(create(clusters, config).fit_predict(matrix), len(matrix))
        if np.unique(labels).size < 2:
            raise ModelExecutionError(
                f"candidate {clusters} produced fewer than two populated clusters"
            )
        try:
            score = float(scorer(matrix, labels))
        except ValueError as exc:
            raise ModelExecutionError(
                f"cannot score candidate {clusters}: {exc}"
            ) from exc
        if not np.isfinite(score):
            raise ModelExecutionError(f"candidate {clusters} produced a non-finite score")
        scores.append(ClusterScore(clusters=clusters, silhouette=score))
    selected = max(scores, key=lambda item: (item.silhouette, -item.clusters)).clusters
    labels = _labels(create(selected, config).fit_predict(matrix), len(matrix))
    return ClusterResult(
        labels=labels,
        selected_clusters=selected,
        scores=tuple(scores),
    )


def validate_numeric_matrix(data: object) -> FloatMatrix:
    """Coerce and validate a numeric two-dimensional clustering matrix."""
    try:
        matrix = np.asarray(data, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"features must be numeric: {exc}") from exc
    if matrix.ndim != 2:
        raise DataValidationError("features must be a two-dimensional matrix")
    if matrix.shape[0] < 3 or matrix.shape[1] < 1:
        raise DataValidationError("features require at least 3 rows and 1 column")
    if not np.isfinite(matrix).all():
        raise DataValidationError("features contain NaN or infinite values")
    return matrix


def _labels(raw: object, expected: int) -> IntVector:
    labels = np.asarray(raw, dtype=np.int64)
    if labels.ndim != 1 or labels.shape[0] != expected:
        raise ModelExecutionError("cluster model returned malformed labels")
    return labels


def _create_kmeans(clusters: int, config: ClusteringConfig) -> ClusterModel:
    return cast(
        ClusterModel,
        KMeans(
            n_clusters=clusters,
            init="k-means++",
            n_init=config.n_init,
            max_iter=config.max_iter,
            random_state=config.random_state,
        ),
    )

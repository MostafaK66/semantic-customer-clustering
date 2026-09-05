"""K-Prototypes clustering behind optional dependency boundaries."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol, cast

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from semantic_customer_clustering.config import ClusteringConfig
from semantic_customer_clustering.errors import (
    DataValidationError,
    DependencyUnavailableError,
    ModelExecutionError,
)
from semantic_customer_clustering.models import ClusterResult, ClusterScore, IntVector


class MixedModel(Protocol):
    def fit_predict(
        self, data: pd.DataFrame, *, categorical: Sequence[int]
    ) -> object: ...


MixedFactory = Callable[[int, ClusteringConfig], MixedModel]
DistanceFunction = Callable[[object], object]


def categorical_indices(frame: pd.DataFrame, columns: Sequence[str]) -> tuple[int, ...]:
    """Resolve configured categorical names to positional K-Prototypes indices."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DataValidationError(
            f"categorical columns are missing: {', '.join(missing)}"
        )
    resolved: list[int] = []
    for column in columns:
        location = frame.columns.get_loc(column)
        if not isinstance(location, int | np.integer):
            raise DataValidationError(f"column name is not unique: {column}")
        resolved.append(int(location))
    return tuple(resolved)


def cluster_mixed(
    frame: pd.DataFrame,
    categorical: Sequence[int],
    config: ClusteringConfig,
    *,
    factory: MixedFactory | None = None,
    distance: DistanceFunction | None = None,
) -> ClusterResult:
    """Select K-Prototypes cluster count with one reusable Gower matrix."""
    if frame.shape[0] < 3 or frame.shape[1] < 1:
        raise DataValidationError("mixed features require at least 3 rows and 1 column")
    if frame.isnull().any().any():
        raise DataValidationError("mixed features contain missing values")
    invalid = [index for index in categorical if index < 0 or index >= frame.shape[1]]
    if invalid:
        raise DataValidationError("categorical column indices are out of range")
    create = factory or _mixed_factory()
    metric = distance or _gower_distance()
    try:
        matrix = np.asarray(metric(frame.to_numpy()), dtype=np.float64)
    except Exception as exc:
        raise ModelExecutionError(f"cannot calculate Gower distance: {exc}") from exc
    if matrix.shape != (len(frame), len(frame)) or not np.isfinite(matrix).all():
        raise ModelExecutionError("Gower distance returned an invalid square matrix")

    feasible = tuple(value for value in config.mixed_candidates if value < len(frame))
    if not feasible:
        raise DataValidationError("no mixed cluster candidate is feasible")
    scores: list[ClusterScore] = []
    labels_by_cluster: dict[int, IntVector] = {}
    for clusters in feasible:
        try:
            raw = create(clusters, config).fit_predict(frame, categorical=categorical)
            labels = _mixed_labels(raw, len(frame))
            score = float(silhouette_score(matrix, labels, metric="precomputed"))
        except ValueError as exc:
            raise ModelExecutionError(
                f"cannot evaluate mixed candidate {clusters}: {exc}"
            ) from exc
        if not np.isfinite(score):
            raise ModelExecutionError(
                f"mixed candidate {clusters} produced a non-finite score"
            )
        labels_by_cluster[clusters] = labels
        scores.append(ClusterScore(clusters, score))
    selected = max(scores, key=lambda item: (item.silhouette, -item.clusters)).clusters
    return ClusterResult(labels_by_cluster[selected], selected, tuple(scores))


def _mixed_labels(raw: object, expected: int) -> IntVector:
    labels = np.asarray(raw, dtype=np.int64)
    if labels.ndim != 1 or labels.shape[0] != expected:
        raise ModelExecutionError("K-Prototypes returned malformed labels")
    if np.unique(labels).size < 2:
        raise ModelExecutionError("K-Prototypes produced fewer than two clusters")
    return labels


def _mixed_factory() -> MixedFactory:
    try:
        from kmodes.kprototypes import KPrototypes
    except ImportError as exc:
        raise DependencyUnavailableError(
            "mixed clustering requires: pip install 'semantic-customer-clustering[mixed]'"
        ) from exc

    def create(clusters: int, config: ClusteringConfig) -> MixedModel:
        return cast(
            MixedModel,
            KPrototypes(
                n_clusters=clusters,
                n_init=config.n_init,
                max_iter=config.max_iter,
                gamma=config.mixed_gamma,
                init=config.mixed_init,
                random_state=config.random_state,
                n_jobs=-1,
            ),
        )

    return create


def _gower_distance() -> DistanceFunction:
    try:
        import gower
    except ImportError as exc:
        raise DependencyUnavailableError(
            "mixed clustering requires: pip install 'semantic-customer-clustering[mixed]'"
        ) from exc
    return cast(DistanceFunction, gower.gower_matrix)

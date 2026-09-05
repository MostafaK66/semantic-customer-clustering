from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

from semantic_customer_clustering.clustering import (
    ClusterModel,
    cluster_kmeans,
    validate_numeric_matrix,
)
from semantic_customer_clustering.config import ClusteringConfig
from semantic_customer_clustering.errors import DataValidationError, ModelExecutionError

DATA = np.asarray([[0.0], [0.1], [0.2], [10.0], [10.1], [10.2]], dtype=np.float64)


class FakeClusterModel:
    def __init__(self, labels: object, error: Exception | None = None) -> None:
        self.labels = labels
        self.error = error

    def fit_predict(self, data: np.ndarray) -> object:
        del data
        if self.error:
            raise self.error
        return self.labels


def factory_for(labels: object) -> Callable[[int, ClusteringConfig], ClusterModel]:
    return lambda clusters, config: FakeClusterModel(labels)


def test_cluster_kmeans_selects_best_and_refits() -> None:
    calls: list[int] = []

    def factory(clusters: int, config: ClusteringConfig) -> FakeClusterModel:
        del config
        calls.append(clusters)
        labels = [0, 0, 0, 1, 1, 1] if clusters == 2 else [0, 0, 1, 2, 2, 2]
        return FakeClusterModel(labels)

    scores = {2: 0.8, 3: 0.5}

    def score(data: np.ndarray, labels: np.ndarray) -> float:
        del data
        return scores[len(np.unique(labels))]

    result = cluster_kmeans(
        DATA,
        ClusteringConfig(candidates=(2, 3)),
        factory=factory,
        scorer=score,
    )
    assert result.selected_clusters == 2
    assert result.labels.tolist() == [0, 0, 0, 1, 1, 1]
    assert [item.clusters for item in result.scores] == [2, 3]
    assert calls == [2, 3, 2]


def test_cluster_kmeans_tie_selects_fewer_clusters() -> None:
    def factory(clusters: int, config: ClusteringConfig) -> FakeClusterModel:
        del config
        labels = [0, 0, 0, 1, 1, 1] if clusters == 2 else [0, 0, 1, 2, 2, 2]
        return FakeClusterModel(labels)

    result = cluster_kmeans(
        DATA,
        ClusteringConfig(candidates=(3, 2)),
        factory=factory,
        scorer=lambda data, labels: 0.5,
    )
    assert result.selected_clusters == 2


def test_cluster_kmeans_ignores_infeasible_candidates() -> None:
    seen: list[int] = []

    def factory(clusters: int, config: ClusteringConfig) -> FakeClusterModel:
        del config
        seen.append(clusters)
        return FakeClusterModel([0, 0, 0, 1, 1, 1])

    cluster_kmeans(
        DATA,
        ClusteringConfig(candidates=(2, 6)),
        factory=factory,
        scorer=lambda data, labels: 0.5,
    )
    assert seen == [2, 2]


def test_real_kmeans_smoke() -> None:
    result = cluster_kmeans(DATA, ClusteringConfig(candidates=(2,), n_init=2))
    assert result.selected_clusters == 2
    assert len(result.labels) == len(DATA)


@pytest.mark.parametrize(
    "data",
    [
        [1, 2, 3],
        [[1], [2]],
        np.empty((3, 0)),
        [[1], [2], [np.nan]],
        [[1], [2], [np.inf]],
        [["a"], ["b"], ["c"]],
    ],
)
def test_validate_numeric_matrix_rejects_invalid_data(data: object) -> None:
    with pytest.raises(DataValidationError):
        validate_numeric_matrix(data)


def test_cluster_rejects_no_feasible_candidate() -> None:
    with pytest.raises(DataValidationError, match="feasible"):
        cluster_kmeans(DATA[:3], ClusteringConfig(candidates=(3, 4)))


@pytest.mark.parametrize("labels", [[0] * 6, [0, 1], [[0], [0], [0], [1], [1], [1]]])
def test_cluster_rejects_bad_labels(labels: object) -> None:
    with pytest.raises(ModelExecutionError):
        cluster_kmeans(
            DATA,
            ClusteringConfig(candidates=(2,)),
            factory=factory_for(labels),
            scorer=lambda data, values: 0.5,
        )


def test_cluster_wraps_scoring_error() -> None:
    def broken(data: np.ndarray, labels: np.ndarray) -> float:
        del data, labels
        raise ValueError("bad metric")

    with pytest.raises(ModelExecutionError, match="bad metric"):
        cluster_kmeans(
            DATA,
            ClusteringConfig(candidates=(2,)),
            factory=factory_for([0, 0, 0, 1, 1, 1]),
            scorer=broken,
        )


def test_cluster_rejects_nonfinite_score() -> None:
    with pytest.raises(ModelExecutionError, match="non-finite"):
        cluster_kmeans(
            DATA,
            ClusteringConfig(candidates=(2,)),
            factory=factory_for([0, 0, 0, 1, 1, 1]),
            scorer=lambda data, labels: np.nan,
        )

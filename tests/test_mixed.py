from __future__ import annotations

import builtins
import sys
from collections.abc import Sequence

import numpy as np
import pandas as pd
import pytest

from semantic_customer_clustering.config import ClusteringConfig
from semantic_customer_clustering.errors import (
    DataValidationError,
    DependencyUnavailableError,
    ModelExecutionError,
)
from semantic_customer_clustering.mixed import categorical_indices, cluster_mixed


@pytest.fixture
def mixed_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "number": [0.0, 0.1, 0.2, 10.0, 10.1, 10.2],
            "kind": ["a", "a", "a", "b", "b", "b"],
        }
    )


class FakeMixedModel:
    def __init__(self, labels: object, error: Exception | None = None) -> None:
        self.labels = labels
        self.error = error

    def fit_predict(self, data: pd.DataFrame, *, categorical: Sequence[int]) -> object:
        del data, categorical
        if self.error:
            raise self.error
        return self.labels


def distances(values: object) -> np.ndarray:
    array = np.asarray(values)
    numeric = array[:, 0].astype(float)
    result = np.abs(numeric[:, None] - numeric[None, :])
    return result / result.max()


def test_categorical_indices(mixed_frame: pd.DataFrame) -> None:
    assert categorical_indices(mixed_frame, ("kind",)) == (1,)
    with pytest.raises(DataValidationError, match="missing"):
        categorical_indices(mixed_frame, ("unknown",))


def test_categorical_indices_rejects_duplicate_names() -> None:
    frame = pd.DataFrame([[1, 2]], columns=["same", "same"])
    with pytest.raises(DataValidationError, match="not unique"):
        categorical_indices(frame, ("same",))


def test_cluster_mixed_selects_best_and_reuses_distance(
    mixed_frame: pd.DataFrame,
) -> None:
    distance_calls = 0

    def distance(values: object) -> np.ndarray:
        nonlocal distance_calls
        distance_calls += 1
        return distances(values)

    def factory(clusters: int, config: ClusteringConfig) -> FakeMixedModel:
        del config
        labels = [0, 0, 0, 1, 1, 1] if clusters == 2 else [0, 0, 1, 2, 2, 2]
        return FakeMixedModel(labels)

    result = cluster_mixed(
        mixed_frame,
        (1,),
        ClusteringConfig(mixed_candidates=(2, 3)),
        factory=factory,
        distance=distance,
    )
    assert result.selected_clusters == 2
    assert len(result.scores) == 2
    assert distance_calls == 1


@pytest.mark.parametrize(
    "frame",
    [pd.DataFrame(), pd.DataFrame({"a": [1, 2]}), pd.DataFrame({"a": [1, 2, None]})],
)
def test_cluster_mixed_rejects_invalid_frames(frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError):
        cluster_mixed(
            frame, (), ClusteringConfig(), factory=lambda c, x: FakeMixedModel([])
        )


def test_cluster_mixed_rejects_invalid_category_index(
    mixed_frame: pd.DataFrame,
) -> None:
    with pytest.raises(DataValidationError, match="out of range"):
        cluster_mixed(
            mixed_frame,
            (2,),
            ClusteringConfig(),
            factory=lambda c, x: FakeMixedModel([]),
        )


@pytest.mark.parametrize(
    "distance",
    [
        lambda values: np.ones((2, 2)),
        lambda values: np.full((6, 6), np.nan),
    ],
)
def test_cluster_mixed_rejects_bad_distance(
    mixed_frame: pd.DataFrame, distance: object
) -> None:
    with pytest.raises(ModelExecutionError, match="invalid"):
        cluster_mixed(
            mixed_frame,
            (1,),
            ClusteringConfig(),
            factory=lambda c, x: FakeMixedModel([0, 0, 0, 1, 1, 1]),
            distance=distance,  # type: ignore[arg-type]
        )


def test_cluster_mixed_wraps_distance_error(mixed_frame: pd.DataFrame) -> None:
    def broken(values: object) -> object:
        del values
        raise RuntimeError("distance failed")

    with pytest.raises(ModelExecutionError, match="distance failed"):
        cluster_mixed(
            mixed_frame,
            (1,),
            ClusteringConfig(),
            factory=lambda c, x: FakeMixedModel([]),
            distance=broken,
        )


def test_cluster_mixed_rejects_no_feasible_candidate(
    mixed_frame: pd.DataFrame,
) -> None:
    with pytest.raises(DataValidationError, match="feasible"):
        cluster_mixed(
            mixed_frame,
            (1,),
            ClusteringConfig(mixed_candidates=(6, 7)),
            factory=lambda c, x: FakeMixedModel([]),
            distance=distances,
        )


@pytest.mark.parametrize(
    "model",
    [
        FakeMixedModel([0, 1]),
        FakeMixedModel([0, 0, 0, 0, 0, 0]),
        FakeMixedModel([], ValueError("fit failed")),
    ],
)
def test_cluster_mixed_rejects_bad_model_output(
    mixed_frame: pd.DataFrame, model: FakeMixedModel
) -> None:
    with pytest.raises(ModelExecutionError):
        cluster_mixed(
            mixed_frame,
            (1,),
            ClusteringConfig(mixed_candidates=(2,)),
            factory=lambda c, x: model,
            distance=distances,
        )


def test_cluster_mixed_rejects_nonfinite_score(
    mixed_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "semantic_customer_clustering.mixed.silhouette_score", lambda *a, **k: np.nan
    )
    with pytest.raises(ModelExecutionError, match="non-finite"):
        cluster_mixed(
            mixed_frame,
            (1,),
            ClusteringConfig(mixed_candidates=(2,)),
            factory=lambda c, x: FakeMixedModel([0, 0, 0, 1, 1, 1]),
            distance=distances,
        )


def test_cluster_mixed_explains_missing_dependencies(
    mixed_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def missing(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("kmodes"):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", missing)
    sys.modules.pop("kmodes.kprototypes", None)
    with pytest.raises(DependencyUnavailableError, match="mixed"):
        cluster_mixed(mixed_frame, (1,), ClusteringConfig(), distance=distances)


def test_cluster_mixed_explains_missing_gower(
    mixed_frame: pd.DataFrame, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def missing(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "gower":
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(DependencyUnavailableError, match="mixed"):
        cluster_mixed(
            mixed_frame,
            (1,),
            ClusteringConfig(),
            factory=lambda c, x: FakeMixedModel([0, 0, 0, 1, 1, 1]),
        )

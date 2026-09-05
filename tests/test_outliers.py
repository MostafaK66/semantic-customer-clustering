from __future__ import annotations

import builtins

import numpy as np
import pytest

from semantic_customer_clustering.errors import (
    DependencyUnavailableError,
    ModelExecutionError,
)
from semantic_customer_clustering.outliers import (
    EcodOutlierDetector,
    IsolationForestOutlierDetector,
)

DATA = np.ones((5, 2), dtype=np.float64)


class FakeOutlierModel:
    def __init__(self, labels: object, error: Exception | None = None) -> None:
        self.labels = labels
        self.error = error

    def fit_predict(self, data: np.ndarray) -> object:
        del data
        if self.error:
            raise self.error
        return self.labels


def test_isolation_forest_adapter_returns_inlier_mask() -> None:
    detector = IsolationForestOutlierDetector(
        0.2, model=FakeOutlierModel([1, 1, -1, 1, 1])
    )
    assert detector.inlier_mask(DATA).tolist() == [True, True, False, True, True]


def test_isolation_forest_real_smoke() -> None:
    values = np.asarray([[float(index)] for index in range(20)], dtype=np.float64)
    mask = IsolationForestOutlierDetector(0.1, random_state=7).inlier_mask(values)
    assert mask.dtype == np.bool_
    assert np.count_nonzero(mask) == 18


@pytest.mark.parametrize(
    ("model", "message"),
    [
        (FakeOutlierModel([], RuntimeError("failed")), "failed"),
        (FakeOutlierModel([1, 1]), "malformed"),
        (FakeOutlierModel([1, 1, 0, 1, 1]), "-1 and 1"),
        (FakeOutlierModel([-1, -1, -1, 1, 1]), "fewer than three"),
    ],
)
def test_isolation_forest_rejects_model_failures(
    model: FakeOutlierModel, message: str
) -> None:
    with pytest.raises(ModelExecutionError, match=message):
        IsolationForestOutlierDetector(0.1, model=model).inlier_mask(DATA)


def test_ecod_adapter_returns_inlier_mask() -> None:
    contamination: list[float] = []

    def factory(value: float) -> FakeOutlierModel:
        contamination.append(value)
        return FakeOutlierModel([0, 0, 1, 0, 0])

    mask = EcodOutlierDetector(0.2, factory=factory).inlier_mask(DATA)
    assert mask.tolist() == [True, True, False, True, True]
    assert contamination == [0.2]


def test_ecod_adapter_wraps_model_failure() -> None:
    detector = EcodOutlierDetector(
        0.1, factory=lambda _: FakeOutlierModel([], RuntimeError("failed"))
    )
    with pytest.raises(ModelExecutionError, match="failed"):
        detector.inlier_mask(DATA)


@pytest.mark.parametrize(
    ("labels", "message"),
    [
        ([0, 1], "malformed"),
        ([[0], [0], [0], [0], [0]], "malformed"),
        ([0, 0, 0, 0, 2], "only 0 and 1"),
        ([1, 1, 1, 0, 0], "fewer than three"),
    ],
)
def test_ecod_adapter_rejects_bad_labels(labels: object, message: str) -> None:
    detector = EcodOutlierDetector(0.1, factory=lambda _: FakeOutlierModel(labels))
    with pytest.raises(ModelExecutionError, match=message):
        detector.inlier_mask(DATA)


def test_ecod_adapter_explains_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("pyod"):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(DependencyUnavailableError, match="outliers"):
        EcodOutlierDetector(0.1).inlier_mask(DATA)

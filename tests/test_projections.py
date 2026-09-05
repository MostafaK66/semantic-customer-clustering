from __future__ import annotations

import numpy as np
import pytest

from semantic_customer_clustering.config import ProjectionConfig
from semantic_customer_clustering.errors import DataValidationError, ModelExecutionError
from semantic_customer_clustering.projections import project_pca, project_tsne

DATA = np.asarray([[float(i), float(i * i)] for i in range(10)], dtype=np.float64)
LABELS = np.asarray([0] * 5 + [1] * 5, dtype=np.int64)


class FakeProjector:
    def __init__(
        self,
        components: int = 2,
        result: object | None = None,
        error: Exception | None = None,
    ) -> None:
        self.components = components
        self.result = result
        self.error = error
        self.seen: np.ndarray | None = None

    def fit_transform(self, data: np.ndarray) -> object:
        self.seen = data.copy()
        if self.error:
            raise self.error
        if self.result is not None:
            return self.result
        return np.zeros((len(data), self.components))


def test_project_pca_real_smoke() -> None:
    result = project_pca(DATA, LABELS, ProjectionConfig(components=2))
    assert result.frame.shape == (10, 3)
    assert result.sampled_indices == tuple(range(10))
    assert result.frame["cluster"].tolist() == LABELS.tolist()


def test_project_pca_rejects_too_many_components() -> None:
    with pytest.raises(DataValidationError, match="components"):
        project_pca(DATA[:, :1], LABELS, ProjectionConfig(components=2))


@pytest.mark.parametrize("labels", [[0, 1], [[0]] * 10])
def test_projections_reject_misaligned_labels(labels: object) -> None:
    with pytest.raises(DataValidationError, match="aligned"):
        project_pca(DATA, labels, ProjectionConfig(components=2))


def test_project_tsne_samples_features_and_labels_together() -> None:
    projector = FakeProjector()
    config = ProjectionConfig(
        components=2,
        sample_fraction=0.5,
        tsne_perplexity=30,
        tsne_max_iter=250,
        random_state=8,
    )
    result = project_tsne(DATA, LABELS, config, projector=projector)
    assert len(result.sampled_indices) == 5
    assert len(set(result.sampled_indices)) == 5
    assert projector.seen is not None
    assert projector.seen.tolist() == DATA[list(result.sampled_indices)].tolist()
    assert (
        result.frame["cluster"].tolist() == LABELS[list(result.sampled_indices)].tolist()
    )


def test_project_tsne_uses_all_rows() -> None:
    result = project_tsne(
        DATA,
        LABELS,
        ProjectionConfig(components=2, sample_fraction=1),
        projector=FakeProjector(),
    )
    assert result.sampled_indices == tuple(range(10))


def test_project_tsne_real_smoke_caps_perplexity() -> None:
    result = project_tsne(
        DATA[:5],
        LABELS[:5],
        ProjectionConfig(
            components=2,
            tsne_perplexity=200,
            tsne_max_iter=250,
            sample_fraction=1,
        ),
    )
    assert result.frame.shape == (5, 3)


@pytest.mark.parametrize(
    "projector",
    [
        FakeProjector(error=ValueError("broken")),
        FakeProjector(result=[[1.0]]),
        FakeProjector(result=np.full((10, 2), np.nan)),
    ],
)
def test_projection_rejects_projector_failures(projector: FakeProjector) -> None:
    with pytest.raises(ModelExecutionError):
        project_pca(DATA, LABELS, ProjectionConfig(components=2), projector=projector)

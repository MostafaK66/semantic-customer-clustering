"""Row-aligned PCA and t-SNE projections."""

from __future__ import annotations

from typing import Protocol

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from semantic_customer_clustering.clustering import validate_numeric_matrix
from semantic_customer_clustering.config import ProjectionConfig
from semantic_customer_clustering.errors import DataValidationError, ModelExecutionError
from semantic_customer_clustering.models import FloatMatrix, IntVector, ProjectionResult


class Projector(Protocol):
    def fit_transform(self, data: FloatMatrix) -> object: ...


def project_pca(
    data: object,
    labels: object,
    config: ProjectionConfig,
    *,
    projector: Projector | None = None,
) -> ProjectionResult:
    """Project every row with PCA and retain cluster alignment."""
    matrix = validate_numeric_matrix(data)
    cluster_labels = _aligned_labels(labels, len(matrix))
    if config.components > min(matrix.shape):
        raise DataValidationError(
            "PCA components cannot exceed the smaller input dimension"
        )
    model = projector or PCA(n_components=config.components)
    projected = _projection_matrix(model, matrix, config.components, "PCA")
    return _result(projected, cluster_labels, np.arange(len(matrix)))


def project_tsne(
    data: object,
    labels: object,
    config: ProjectionConfig,
    *,
    projector: Projector | None = None,
) -> ProjectionResult:
    """Sample rows once, preserving label alignment, and fit t-SNE once."""
    matrix = validate_numeric_matrix(data)
    cluster_labels = _aligned_labels(labels, len(matrix))
    sample_size = max(3, int(np.ceil(len(matrix) * config.sample_fraction)))
    sample_size = min(sample_size, len(matrix))
    if sample_size == len(matrix):
        positions: npt.NDArray[np.int64] = np.arange(len(matrix), dtype=np.int64)
    else:
        generator = np.random.default_rng(config.random_state)
        positions = np.asarray(
            np.sort(generator.choice(len(matrix), sample_size, replace=False)),
            dtype=np.int64,
        )
    sampled = matrix[positions]
    perplexity = min(config.tsne_perplexity, float(sample_size - 1))
    model = projector or TSNE(
        n_components=config.components,
        learning_rate="auto",
        init="random",
        perplexity=perplexity,
        max_iter=config.tsne_max_iter,
        random_state=config.random_state,
    )
    projected = _projection_matrix(model, sampled, config.components, "t-SNE")
    return _result(projected, cluster_labels[positions], positions)


def _aligned_labels(raw: object, expected: int) -> IntVector:
    labels = np.asarray(raw, dtype=np.int64)
    if labels.ndim != 1 or len(labels) != expected:
        raise DataValidationError("cluster labels are not aligned with feature rows")
    return labels


def _projection_matrix(
    model: Projector,
    data: FloatMatrix,
    components: int,
    name: str,
) -> FloatMatrix:
    try:
        projected = np.asarray(model.fit_transform(data), dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ModelExecutionError(f"{name} projection failed: {exc}") from exc
    if projected.shape != (len(data), components) or not np.isfinite(projected).all():
        raise ModelExecutionError(f"{name} returned an invalid projection matrix")
    return projected


def _result(
    projected: FloatMatrix,
    labels: IntVector,
    positions: IntVector,
) -> ProjectionResult:
    columns = [f"component_{index + 1}" for index in range(projected.shape[1])]
    frame = pd.DataFrame(projected, columns=columns)
    frame["cluster"] = labels
    return ProjectionResult(frame=frame, sampled_indices=tuple(positions.tolist()))

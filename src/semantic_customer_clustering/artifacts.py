"""Filesystem artifact persistence kept separate from model computation."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from semantic_customer_clustering.errors import DataValidationError, ModelExecutionError
from semantic_customer_clustering.models import (
    ClusterResult,
    PipelineArtifacts,
    ProjectionResult,
)


def save_artifacts(
    output_dir: Path,
    prefix: str,
    source_indices: Sequence[object],
    result: ClusterResult,
    pca: ProjectionResult,
    tsne: ProjectionResult | None,
) -> PipelineArtifacts:
    """Write row assignments, candidate scores, and projections as CSV."""
    if not prefix or any(character in prefix for character in ("/", "\\")):
        raise DataValidationError("artifact prefix must be a simple non-empty name")
    if len(source_indices) != len(result.labels):
        raise DataValidationError("source indices do not align with cluster labels")
    assignments_path = output_dir / f"{prefix}_assignments.csv"
    scores_path = output_dir / f"{prefix}_scores.csv"
    pca_path = output_dir / f"{prefix}_pca.csv"
    tsne_path = output_dir / f"{prefix}_tsne.csv" if tsne is not None else None
    try:
        output_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            {"source_index": list(source_indices), "cluster": result.labels}
        ).to_csv(assignments_path, index=False)
        pd.DataFrame(
            [
                {"clusters": score.clusters, "silhouette": score.silhouette}
                for score in result.scores
            ]
        ).to_csv(scores_path, index=False)
        _with_source_indices(pca, source_indices).to_csv(pca_path, index=False)
        if tsne is not None and tsne_path is not None:
            _with_source_indices(tsne, source_indices).to_csv(tsne_path, index=False)
    except OSError as exc:
        raise ModelExecutionError(
            f"cannot write artifacts to {output_dir}: {exc}"
        ) from exc
    return PipelineArtifacts(assignments_path, scores_path, pca_path, tsne_path)


def _with_source_indices(
    projection: ProjectionResult,
    source_indices: Sequence[object],
) -> pd.DataFrame:
    try:
        selected = [source_indices[position] for position in projection.sampled_indices]
    except (IndexError, TypeError, ValueError) as exc:
        raise DataValidationError(
            "projection indices do not align with input rows"
        ) from exc
    frame = projection.frame.copy()
    if len(frame) != len(selected):
        raise DataValidationError("projection rows do not align with sampled indices")
    frame.insert(0, "source_index", selected)
    return frame

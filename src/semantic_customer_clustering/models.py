"""Immutable data transferred between clustering stages."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd

FloatMatrix = npt.NDArray[np.float64]
IntVector = npt.NDArray[np.int64]
BoolVector = npt.NDArray[np.bool_]


@dataclass(frozen=True, slots=True)
class ClusterScore:
    """Silhouette score for one candidate cluster count."""

    clusters: int
    silhouette: float


@dataclass(frozen=True, slots=True)
class ClusterResult:
    """Selected model output and all candidate scores."""

    labels: IntVector
    selected_clusters: int
    scores: tuple[ClusterScore, ...]


@dataclass(frozen=True, slots=True)
class ProjectionResult:
    """A row-aligned lower-dimensional representation."""

    frame: pd.DataFrame
    sampled_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class PipelineArtifacts:
    """Paths produced by a completed pipeline run."""

    assignments: Path
    scores: Path
    pca: Path
    tsne: Path | None

"""Optional static plotting adapter."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from semantic_customer_clustering.errors import (
    DependencyUnavailableError,
    ModelExecutionError,
)
from semantic_customer_clustering.models import ProjectionResult


def save_projection_plot(projection: ProjectionResult, path: Path, title: str) -> Path:
    """Save a two- or three-dimensional cluster plot without opening a GUI."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise DependencyUnavailableError(
            "plotting requires: pip install 'semantic-customer-clustering[plots]'"
        ) from exc
    dimensions = [
        column for column in projection.frame.columns if column.startswith("component_")
    ]
    if len(dimensions) not in (2, 3):
        raise ModelExecutionError(
            "plotting requires a two- or three-dimensional projection"
        )
    figure: Any = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d" if len(dimensions) == 3 else None)
    clusters = projection.frame["cluster"]
    scatter_args = [projection.frame[column] for column in dimensions]
    axis.scatter(*scatter_args, c=clusters, cmap="viridis", s=18, alpha=0.8)
    axis.set_title(title)
    for index, column in enumerate(dimensions):
        setter = (axis.set_xlabel, axis.set_ylabel, getattr(axis, "set_zlabel", None))[
            index
        ]
        if setter is not None:
            setter(column)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.tight_layout()
        figure.savefig(path)
    except OSError as exc:
        raise ModelExecutionError(f"cannot save plot {path}: {exc}") from exc
    finally:
        plt.close(figure)
    return path

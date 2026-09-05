from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semantic_customer_clustering.artifacts import save_artifacts
from semantic_customer_clustering.errors import DataValidationError, ModelExecutionError
from semantic_customer_clustering.models import (
    ClusterResult,
    ClusterScore,
    ProjectionResult,
)


def cluster_result() -> ClusterResult:
    return ClusterResult(
        np.asarray([0, 1, 1], dtype=np.int64),
        2,
        (ClusterScore(2, 0.7),),
    )


def projection(indices: tuple[int, ...] = (0, 1, 2)) -> ProjectionResult:
    return ProjectionResult(
        pd.DataFrame(
            {
                "component_1": [0.0] * len(indices),
                "component_2": [1.0] * len(indices),
                "cluster": [0, 1, 1][: len(indices)],
            }
        ),
        indices,
    )


def test_save_artifacts_writes_aligned_csv_files(tmp_path: Path) -> None:
    result = save_artifacts(
        tmp_path / "nested",
        "test",
        (10, 20, 30),
        cluster_result(),
        projection(),
        projection((0, 2)),
    )
    assert result.assignments.exists()
    assert result.scores.exists()
    assert result.pca.exists()
    assert result.tsne is not None and result.tsne.exists()
    assert pd.read_csv(result.tsne)["source_index"].tolist() == [10, 30]


def test_save_artifacts_can_skip_tsne(tmp_path: Path) -> None:
    result = save_artifacts(
        tmp_path, "test", (0, 1, 2), cluster_result(), projection(), None
    )
    assert result.tsne is None


@pytest.mark.parametrize("prefix", ["", "bad/name", "bad\\name"])
def test_save_artifacts_rejects_unsafe_prefix(tmp_path: Path, prefix: str) -> None:
    with pytest.raises(DataValidationError, match="prefix"):
        save_artifacts(tmp_path, prefix, (0, 1, 2), cluster_result(), projection(), None)


def test_save_artifacts_rejects_source_label_mismatch(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="source indices"):
        save_artifacts(tmp_path, "x", (0,), cluster_result(), projection(), None)


def test_save_artifacts_rejects_projection_index_mismatch(tmp_path: Path) -> None:
    with pytest.raises(DataValidationError, match="projection indices"):
        save_artifacts(
            tmp_path,
            "x",
            (0, 1, 2),
            cluster_result(),
            projection((5,)),
            None,
        )


def test_save_artifacts_rejects_projection_row_mismatch(tmp_path: Path) -> None:
    bad = ProjectionResult(projection().frame.iloc[:1], (0, 1))
    with pytest.raises(DataValidationError, match="projection rows"):
        save_artifacts(tmp_path, "x", (0, 1, 2), cluster_result(), bad, None)


def test_save_artifacts_wraps_filesystem_error(tmp_path: Path) -> None:
    file_path = tmp_path / "not-a-directory"
    file_path.write_text("occupied", encoding="utf-8")
    with pytest.raises(ModelExecutionError, match="cannot write"):
        save_artifacts(file_path, "x", (0, 1, 2), cluster_result(), projection(), None)

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from semantic_customer_clustering.config import AppConfig
from semantic_customer_clustering.models import (
    ClusterResult,
    ClusterScore,
    ProjectionResult,
)
from semantic_customer_clustering.outliers import IsolationForestOutlierDetector
from semantic_customer_clustering.pipeline import (
    default_detector,
    run_classical,
    run_mixed,
    run_semantic,
)


def reader_for(frame: pd.DataFrame):
    def reader(path: object, *, sep: object) -> pd.DataFrame:
        del path, sep
        return frame

    return reader


class FakeEncoder:
    def __init__(self) -> None:
        self.texts: Sequence[str] = ()

    def encode(self, sentences: Sequence[str]) -> np.ndarray:
        self.texts = sentences
        half = len(sentences) // 2
        return np.asarray(
            [
                [float(index < half), float(index >= half)]
                for index in range(len(sentences))
            ],
            dtype=np.float64,
        )


class FakeDetector:
    def inlier_mask(self, data: np.ndarray) -> np.ndarray:
        mask = np.ones(len(data), dtype=np.bool_)
        mask[0] = False
        return mask


def test_run_classical_writes_outputs(
    app_config: AppConfig, customer_frame: pd.DataFrame
) -> None:
    artifacts = run_classical(
        app_config,
        reader=reader_for(customer_frame),
        detector=FakeDetector(),
        include_tsne=False,
    )
    assert artifacts.assignments.exists()
    assignments = pd.read_csv(artifacts.assignments)
    assert len(assignments) == len(customer_frame) - 1
    assert assignments["source_index"].iloc[0] == 101
    assert artifacts.tsne is None


def test_run_semantic_uses_injected_encoder(
    app_config: AppConfig, customer_frame: pd.DataFrame
) -> None:
    encoder = FakeEncoder()
    artifacts = run_semantic(
        app_config,
        reader=reader_for(customer_frame),
        encoder=encoder,
        include_tsne=False,
    )
    assert len(encoder.texts) == len(customer_frame)
    assert encoder.texts[0].startswith("Age: 21")
    assert artifacts.assignments.name == "semantic_assignments.csv"


def test_run_mixed_uses_categorical_indices_and_numeric_projection(
    app_config: AppConfig,
    customer_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[tuple[int, ...]] = []

    def fake_cluster(
        frame: pd.DataFrame, indices: tuple[int, ...], config: object
    ) -> ClusterResult:
        del config
        seen.append(indices)
        labels = np.asarray([0] * 6 + [1] * 6, dtype=np.int64)
        return ClusterResult(labels, 2, (ClusterScore(2, 0.8),))

    monkeypatch.setattr(
        "semantic_customer_clustering.pipeline.cluster_mixed", fake_cluster
    )
    artifacts = run_mixed(
        app_config,
        reader=reader_for(customer_frame),
        include_tsne=False,
    )
    assert seen == [(1, 2, 3, 4, 6, 7)]
    assert artifacts.assignments.name == "mixed_assignments.csv"


def test_pipeline_can_save_plots(
    app_config: AppConfig,
    customer_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths: list[Path] = []

    def fake_plot(projection: object, path: Path, title: str) -> Path:
        del projection, title
        paths.append(path)
        return path

    monkeypatch.setattr(
        "semantic_customer_clustering.pipeline.save_projection_plot", fake_plot
    )
    run_classical(
        app_config,
        reader=reader_for(customer_frame),
        include_tsne=False,
        plots=True,
    )
    assert [path.name for path in paths] == ["classical_pca.png"]


def test_pipeline_saves_tsne_plot(
    app_config: AppConfig,
    customer_frame: pd.DataFrame,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths: list[Path] = []

    def fake_tsne(data: object, labels: object, config: object) -> ProjectionResult:
        del data, config
        values = np.asarray(labels)
        return ProjectionResult(
            pd.DataFrame(
                {
                    "component_1": np.zeros(len(values)),
                    "component_2": np.ones(len(values)),
                    "cluster": values,
                }
            ),
            tuple(range(len(values))),
        )

    monkeypatch.setattr("semantic_customer_clustering.pipeline.project_tsne", fake_tsne)
    monkeypatch.setattr(
        "semantic_customer_clustering.pipeline.save_projection_plot",
        lambda projection, path, title: paths.append(path) or path,
    )
    artifacts = run_classical(
        app_config,
        reader=reader_for(customer_frame),
        include_tsne=True,
        plots=True,
    )
    assert artifacts.tsne is not None
    assert [path.name for path in paths] == ["classical_pca.png", "classical_tsne.png"]


def test_default_detector_uses_config(app_config: AppConfig) -> None:
    detector = default_detector(app_config)
    assert isinstance(detector, IsolationForestOutlierDetector)

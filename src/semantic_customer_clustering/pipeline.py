"""Application workflows that compose pure logic with injected boundaries."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pandas as pd

from semantic_customer_clustering.artifacts import save_artifacts
from semantic_customer_clustering.clustering import cluster_kmeans
from semantic_customer_clustering.config import AppConfig
from semantic_customer_clustering.data import (
    CsvReader,
    compile_customer_texts,
    prepare_classical_features,
    read_customer_data,
)
from semantic_customer_clustering.embedding import SentenceTransformerEncoder
from semantic_customer_clustering.mixed import categorical_indices, cluster_mixed
from semantic_customer_clustering.models import (
    BoolVector,
    ClusterResult,
    FloatMatrix,
    PipelineArtifacts,
)
from semantic_customer_clustering.outliers import IsolationForestOutlierDetector
from semantic_customer_clustering.plotting import save_projection_plot
from semantic_customer_clustering.projections import project_pca, project_tsne


class TextEncoder(Protocol):
    def encode(self, sentences: Sequence[str]) -> FloatMatrix: ...


class InlierDetector(Protocol):
    def inlier_mask(self, data: FloatMatrix) -> BoolVector: ...


def run_classical(
    config: AppConfig,
    *,
    reader: CsvReader = pd.read_csv,
    detector: InlierDetector | None = None,
    include_tsne: bool = True,
    plots: bool = False,
) -> PipelineArtifacts:
    """Run schema-aware preprocessing followed by K-means clustering."""
    frame = read_customer_data(config.data, reader=reader)
    features, _ = prepare_classical_features(frame, config.data)
    return _run_numeric(
        "classical",
        frame,
        features,
        config,
        detector=detector,
        include_tsne=include_tsne,
        plots=plots,
    )


def run_semantic(
    config: AppConfig,
    *,
    reader: CsvReader = pd.read_csv,
    encoder: TextEncoder | None = None,
    detector: InlierDetector | None = None,
    include_tsne: bool = True,
    plots: bool = False,
) -> PipelineArtifacts:
    """Embed customer records locally, then cluster their semantic vectors."""
    frame = read_customer_data(config.data, reader=reader)
    texts = compile_customer_texts(frame, config.data.columns)
    selected_encoder = encoder or SentenceTransformerEncoder(config.semantic)
    features = selected_encoder.encode(texts)
    return _run_numeric(
        "semantic",
        frame,
        features,
        config,
        detector=detector,
        include_tsne=include_tsne,
        plots=plots,
    )


def run_mixed(
    config: AppConfig,
    *,
    reader: CsvReader = pd.read_csv,
    detector: InlierDetector | None = None,
    include_tsne: bool = True,
    plots: bool = False,
) -> PipelineArtifacts:
    """Run K-Prototypes while using numeric features only for projection."""
    frame = read_customer_data(config.data, reader=reader)
    numeric_features, _ = prepare_classical_features(frame, config.data)
    frame, numeric_features = _filter(frame, numeric_features, detector)
    indices = categorical_indices(frame, config.data.categorical_columns)
    result = cluster_mixed(frame, indices, config.clustering)
    return _project_and_save(
        "mixed",
        frame,
        numeric_features,
        result,
        config,
        include_tsne,
        plots,
    )


def default_detector(config: AppConfig) -> IsolationForestOutlierDetector:
    """Construct the portable production outlier adapter."""
    return IsolationForestOutlierDetector(
        config.clustering.contamination,
        random_state=config.clustering.random_state,
    )


def _run_numeric(
    prefix: str,
    frame: pd.DataFrame,
    features: FloatMatrix,
    config: AppConfig,
    *,
    detector: InlierDetector | None,
    include_tsne: bool,
    plots: bool,
) -> PipelineArtifacts:
    filtered_frame, filtered_features = _filter(frame, features, detector)
    result = cluster_kmeans(filtered_features, config.clustering)
    return _project_and_save(
        prefix,
        filtered_frame,
        filtered_features,
        result,
        config,
        include_tsne,
        plots,
    )


def _filter(
    frame: pd.DataFrame,
    features: FloatMatrix,
    detector: InlierDetector | None,
) -> tuple[pd.DataFrame, FloatMatrix]:
    if detector is None:
        return frame, features
    mask = np.asarray(detector.inlier_mask(features), dtype=np.bool_)
    return frame.iloc[mask].copy(), features[mask]


def _project_and_save(
    prefix: str,
    frame: pd.DataFrame,
    features: FloatMatrix,
    result: ClusterResult,
    config: AppConfig,
    include_tsne: bool,
    plots: bool,
) -> PipelineArtifacts:
    pca = project_pca(features, result.labels, config.projection)
    tsne = (
        project_tsne(features, result.labels, config.projection) if include_tsne else None
    )
    artifacts = save_artifacts(
        config.output_dir,
        prefix,
        tuple(frame.index),
        result,
        pca,
        tsne,
    )
    if plots:
        save_projection_plot(pca, config.output_dir / f"{prefix}_pca.png", "PCA")
        if tsne is not None:
            save_projection_plot(
                tsne,
                config.output_dir / f"{prefix}_tsne.png",
                "t-SNE",
            )
    return artifacts

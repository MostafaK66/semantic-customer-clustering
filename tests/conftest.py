from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from semantic_customer_clustering.config import (
    AppConfig,
    ClusteringConfig,
    DataConfig,
    ProjectionConfig,
    SemanticConfig,
)


@pytest.fixture
def customer_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "age": [21, 23, 25, 27, 29, 31, 60, 62, 64, 66, 68, 70],
            "job": ["student"] * 6 + ["retired"] * 6,
            "marital": ["single"] * 6 + ["married"] * 6,
            "education": ["secondary", "tertiary"] * 6,
            "default": ["no"] * 12,
            "balance": [10, 20, 15, 30, 25, 35, 900, 950, 1000, 1050, 1100, 1200],
            "housing": ["yes"] * 6 + ["no"] * 6,
            "loan": ["no", "yes"] * 6,
            "ignored": list(range(12)),
        },
        index=[100 + index for index in range(12)],
    )


@pytest.fixture
def app_config(tmp_path: Path) -> AppConfig:
    return AppConfig(
        data=DataConfig(path=tmp_path / "customers.csv"),
        clustering=ClusteringConfig(
            candidates=(2,),
            mixed_candidates=(2,),
            contamination=0.1,
        ),
        projection=ProjectionConfig(
            components=2,
            tsne_perplexity=3,
            tsne_max_iter=250,
            sample_fraction=0.5,
        ),
        semantic=SemanticConfig(show_progress=False),
        output_dir=tmp_path / "output",
    )

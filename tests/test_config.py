from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from semantic_customer_clustering.config import (
    AppConfig,
    ClusteringConfig,
    DataConfig,
    ProjectionConfig,
    SemanticConfig,
)
from semantic_customer_clustering.errors import ConfigurationError


def test_default_config_is_immutable() -> None:
    config = AppConfig()
    assert config.clustering.candidates == (2, 3, 4)
    with pytest.raises(FrozenInstanceError):
        config.output_dir = Path("elsewhere")  # type: ignore[misc]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"delimiter": ""}, "delimiter"),
        ({"columns": ()}, "unique"),
        ({"columns": ("age", "age")}, "unique"),
        ({"columns": ("age",), "categorical_columns": ("job",)}, "not in"),
        (
            {
                "columns": ("age", "job"),
                "categorical_columns": ("job",),
                "ordinal_columns": ("age",),
            },
            "ordinal",
        ),
    ],
)
def test_data_config_rejects_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        DataConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"candidates": ()}, "candidates"),
        ({"candidates": (1,)}, "candidates"),
        ({"candidates": (2, 2)}, "duplicates"),
        ({"mixed_candidates": ()}, "mixed_candidates"),
        ({"mixed_candidates": (1,)}, "mixed_candidates"),
        ({"mixed_candidates": (2, 2)}, "duplicates"),
        ({"n_init": 0}, "positive"),
        ({"max_iter": 0}, "positive"),
        ({"contamination": 0}, "contamination"),
        ({"contamination": 0.5}, "contamination"),
        ({"mixed_gamma": 0}, "mixed_gamma"),
        ({"mixed_init": " "}, "mixed_gamma"),
    ],
)
def test_clustering_config_rejects_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        ClusteringConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"components": 1}, "at least"),
        ({"components": 4}, "exceed"),
        ({"tsne_perplexity": 0}, "perplexity"),
        ({"tsne_max_iter": 249}, "at least"),
        ({"sample_fraction": 0}, "sample_fraction"),
        ({"sample_fraction": 1.1}, "sample_fraction"),
    ],
)
def test_projection_config_rejects_invalid_values(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ConfigurationError, match=message):
        ProjectionConfig(**kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize("kwargs", [{"model_name": " "}, {"batch_size": 0}])
def test_semantic_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ConfigurationError):
        SemanticConfig(**kwargs)  # type: ignore[arg-type]


def test_load_complete_toml_resolves_paths(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        """
output_dir = "artifacts"
[data]
path = "input.csv"
delimiter = ","
columns = ["value", "kind"]
categorical_columns = ["kind"]
ordinal_columns = []
[clustering]
candidates = [2, 4]
mixed_candidates = [2, 3]
n_init = 3
max_iter = 20
random_state = 9
contamination = 0.2
mixed_gamma = 1.5
mixed_init = "Cao"
[projection]
components = 2
tsne_perplexity = 4
tsne_max_iter = 300
sample_fraction = 0.8
random_state = 7
[semantic]
model_name = "local/model"
batch_size = 8
show_progress = false
""",
        encoding="utf-8",
    )
    config = AppConfig.from_toml(path)
    assert config.data.path == tmp_path / "input.csv"
    assert config.output_dir == tmp_path / "artifacts"
    assert config.data.ordinal_columns == ()
    assert config.clustering.candidates == (2, 4)
    assert config.clustering.mixed_init == "Cao"
    assert config.projection.random_state == 7
    assert config.semantic.model_name == "local/model"
    assert config.semantic.show_progress is False


def test_load_minimal_toml_uses_defaults_and_absolute_path(tmp_path: Path) -> None:
    absolute = tmp_path / "absolute.csv"
    path = tmp_path / "minimal.toml"
    path.write_text(f'[data]\npath = "{absolute}"\n', encoding="utf-8")
    config = AppConfig.from_toml(path)
    assert config.data.path == absolute
    assert config.clustering.n_init == 10


@pytest.mark.parametrize(
    "content",
    [
        "not valid toml = [",
        "data = 3",
        "[data]\ncolumns = 4",
        "[clustering]\ncandidates = [true]",
    ],
)
def test_load_toml_rejects_invalid_content(tmp_path: Path, content: str) -> None:
    path = tmp_path / "bad.toml"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ConfigurationError):
        AppConfig.from_toml(path)


def test_load_toml_reports_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read"):
        AppConfig.from_toml(tmp_path / "missing.toml")

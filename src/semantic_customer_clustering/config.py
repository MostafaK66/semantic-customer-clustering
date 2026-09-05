"""Validated, immutable application configuration."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from semantic_customer_clustering.errors import ConfigurationError

DEFAULT_COLUMNS = (
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
)
DEFAULT_CATEGORICAL = (
    "job",
    "marital",
    "education",
    "default",
    "housing",
    "loan",
)


@dataclass(frozen=True, slots=True)
class DataConfig:
    """Input schema and CSV settings."""

    path: Path = Path("data/train_data.csv")
    delimiter: str = ";"
    columns: tuple[str, ...] = DEFAULT_COLUMNS
    categorical_columns: tuple[str, ...] = DEFAULT_CATEGORICAL
    ordinal_columns: tuple[str, ...] = ("education",)

    def __post_init__(self) -> None:
        if not self.delimiter:
            raise ConfigurationError("data.delimiter must not be empty")
        if not self.columns or len(set(self.columns)) != len(self.columns):
            raise ConfigurationError("data.columns must contain unique column names")
        unknown = set(self.categorical_columns) - set(self.columns)
        if unknown:
            joined = ", ".join(sorted(unknown))
            raise ConfigurationError(
                f"categorical columns are not in data.columns: {joined}"
            )
        invalid_ordinal = set(self.ordinal_columns) - set(self.categorical_columns)
        if invalid_ordinal:
            joined = ", ".join(sorted(invalid_ordinal))
            raise ConfigurationError(
                f"ordinal columns are not categorical columns: {joined}"
            )


@dataclass(frozen=True, slots=True)
class ClusteringConfig:
    """Deterministic clustering and outlier settings."""

    candidates: tuple[int, ...] = (2, 3, 4)
    mixed_candidates: tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9)
    n_init: int = 10
    max_iter: int = 100
    random_state: int = 123
    contamination: float = 0.001
    mixed_gamma: float = 2.55
    mixed_init: str = "Huang"

    def __post_init__(self) -> None:
        if not self.candidates or any(value < 2 for value in self.candidates):
            raise ConfigurationError("clustering.candidates must contain integers >= 2")
        if len(set(self.candidates)) != len(self.candidates):
            raise ConfigurationError("clustering.candidates must not contain duplicates")
        if not self.mixed_candidates or any(value < 2 for value in self.mixed_candidates):
            raise ConfigurationError(
                "clustering.mixed_candidates must contain integers >= 2"
            )
        if len(set(self.mixed_candidates)) != len(self.mixed_candidates):
            raise ConfigurationError(
                "clustering.mixed_candidates must not contain duplicates"
            )
        if self.n_init < 1 or self.max_iter < 1:
            raise ConfigurationError("n_init and max_iter must be positive")
        if not 0 < self.contamination < 0.5:
            raise ConfigurationError("contamination must be between 0 and 0.5")
        if self.mixed_gamma <= 0 or not self.mixed_init.strip():
            raise ConfigurationError("mixed_gamma and mixed_init must be valid")


@dataclass(frozen=True, slots=True)
class ProjectionConfig:
    """PCA and t-SNE settings."""

    components: int = 3
    tsne_perplexity: float = 30.0
    tsne_max_iter: int = 1000
    sample_fraction: float = 1.0
    random_state: int = 123

    def __post_init__(self) -> None:
        if self.components < 2:
            raise ConfigurationError("projection.components must be at least 2")
        if self.components > 3:
            raise ConfigurationError("projection.components must not exceed 3")
        if self.tsne_perplexity <= 0:
            raise ConfigurationError("projection.tsne_perplexity must be positive")
        if self.tsne_max_iter < 250:
            raise ConfigurationError("projection.tsne_max_iter must be at least 250")
        if not 0 < self.sample_fraction <= 1:
            raise ConfigurationError("projection.sample_fraction must be in (0, 1]")


@dataclass(frozen=True, slots=True)
class SemanticConfig:
    """Local sentence-embedding model settings."""

    model_name: str = "sentence-transformers/paraphrase-MiniLM-L6-v2"
    batch_size: int = 32
    show_progress: bool = True

    def __post_init__(self) -> None:
        if not self.model_name.strip():
            raise ConfigurationError("semantic.model_name must not be empty")
        if self.batch_size < 1:
            raise ConfigurationError("semantic.batch_size must be positive")


@dataclass(frozen=True, slots=True)
class AppConfig:
    """Complete application configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    output_dir: Path = Path("output")

    @classmethod
    def from_toml(cls, path: Path) -> AppConfig:
        """Load configuration and resolve relative paths beside the TOML file."""
        try:
            with path.open("rb") as stream:
                raw = tomllib.load(stream)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ConfigurationError(f"cannot read configuration {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ConfigurationError("configuration root must be a TOML table")

        base = path.resolve().parent
        data_raw = _table(raw, "data")
        cluster_raw = _table(raw, "clustering")
        projection_raw = _table(raw, "projection")
        semantic_raw = _table(raw, "semantic")
        data_path = _resolve(base, Path(str(data_raw.get("path", "data/train_data.csv"))))
        output_dir = _resolve(base, Path(str(raw.get("output_dir", "output"))))
        return cls(
            data=DataConfig(
                path=data_path,
                delimiter=str(data_raw.get("delimiter", ";")),
                columns=_strings(
                    data_raw.get("columns", DEFAULT_COLUMNS), "data.columns"
                ),
                categorical_columns=_strings(
                    data_raw.get("categorical_columns", DEFAULT_CATEGORICAL),
                    "data.categorical_columns",
                ),
                ordinal_columns=_strings(
                    data_raw.get("ordinal_columns", ("education",)),
                    "data.ordinal_columns",
                ),
            ),
            clustering=ClusteringConfig(
                candidates=_integers(
                    cluster_raw.get("candidates", (2, 3, 4)),
                    "clustering.candidates",
                ),
                mixed_candidates=_integers(
                    cluster_raw.get("mixed_candidates", tuple(range(2, 10))),
                    "clustering.mixed_candidates",
                ),
                n_init=int(cluster_raw.get("n_init", 10)),
                max_iter=int(cluster_raw.get("max_iter", 100)),
                random_state=int(cluster_raw.get("random_state", 123)),
                contamination=float(cluster_raw.get("contamination", 0.001)),
                mixed_gamma=float(cluster_raw.get("mixed_gamma", 2.55)),
                mixed_init=str(cluster_raw.get("mixed_init", "Huang")),
            ),
            projection=ProjectionConfig(
                components=int(projection_raw.get("components", 3)),
                tsne_perplexity=float(projection_raw.get("tsne_perplexity", 30)),
                tsne_max_iter=int(projection_raw.get("tsne_max_iter", 1000)),
                sample_fraction=float(projection_raw.get("sample_fraction", 1)),
                random_state=int(projection_raw.get("random_state", 123)),
            ),
            semantic=SemanticConfig(
                model_name=str(
                    semantic_raw.get(
                        "model_name",
                        "sentence-transformers/paraphrase-MiniLM-L6-v2",
                    )
                ),
                batch_size=int(semantic_raw.get("batch_size", 32)),
                show_progress=bool(semantic_raw.get("show_progress", True)),
            ),
            output_dir=output_dir,
        )


def _table(raw: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = raw.get(key, {})
    if not isinstance(value, dict):
        raise ConfigurationError(f"{key} must be a TOML table")
    return value


def _strings(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list | tuple) or not all(
        isinstance(item, str) for item in value
    ):
        raise ConfigurationError(f"{name} must be an array of strings")
    return tuple(value)


def _integers(value: object, name: str) -> tuple[int, ...]:
    if not isinstance(value, list | tuple) or not all(
        isinstance(item, int) and not isinstance(item, bool) for item in value
    ):
        raise ConfigurationError(f"{name} must be an array of integers")
    return tuple(value)


def _resolve(base: Path, value: Path) -> Path:
    return value if value.is_absolute() else base / value

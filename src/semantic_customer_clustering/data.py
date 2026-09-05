"""CSV loading, schema validation, and deterministic feature preparation."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, PowerTransformer

from semantic_customer_clustering.config import DataConfig
from semantic_customer_clustering.errors import DataValidationError
from semantic_customer_clustering.models import FloatMatrix

CsvReader = Callable[..., pd.DataFrame]


def read_customer_data(
    config: DataConfig,
    *,
    reader: CsvReader = pd.read_csv,
) -> pd.DataFrame:
    """Read a configured CSV through an injectable filesystem boundary."""
    try:
        frame = reader(config.path, sep=config.delimiter)
    except (OSError, pd.errors.ParserError, UnicodeError) as exc:
        raise DataValidationError(f"cannot read input data {config.path}: {exc}") from exc
    return validate_customer_data(frame, config)


def validate_customer_data(frame: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Return a defensive schema-constrained copy of customer data."""
    if frame.empty:
        raise DataValidationError("input data contains no rows")
    missing = [column for column in config.columns if column not in frame.columns]
    if missing:
        raise DataValidationError(f"input data is missing columns: {', '.join(missing)}")
    selected = frame.loc[:, list(config.columns)].copy()
    null_columns = selected.columns[selected.isnull().any()].tolist()
    if null_columns:
        raise DataValidationError(
            f"input data contains missing values in: {', '.join(null_columns)}"
        )
    return selected


def build_preprocessor(config: DataConfig) -> ColumnTransformer:
    """Create the original one-hot, ordinal, and power-transform pipeline."""
    ordinal = list(config.ordinal_columns)
    nominal = [
        column
        for column in config.categorical_columns
        if column not in config.ordinal_columns
    ]
    numeric = [
        column for column in config.columns if column not in config.categorical_columns
    ]
    transformers: list[tuple[str, object, list[str]]] = []
    if nominal:
        transformers.append(
            (
                "nominal",
                Pipeline(
                    [
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="ignore",
                                drop="first",
                                sparse_output=False,
                            ),
                        )
                    ]
                ),
                nominal,
            )
        )
    if ordinal:
        transformers.append(
            (
                "ordinal",
                Pipeline(
                    [
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        )
                    ]
                ),
                ordinal,
            )
        )
    if numeric:
        transformers.append(
            ("numeric", Pipeline([("power", PowerTransformer())]), numeric)
        )
    return ColumnTransformer(transformers=transformers, verbose_feature_names_out=True)


def prepare_classical_features(
    frame: pd.DataFrame,
    config: DataConfig,
    *,
    preprocessor: ColumnTransformer | None = None,
) -> tuple[FloatMatrix, tuple[str, ...]]:
    """Fit preprocessing and return finite numeric features with names."""
    validated = validate_customer_data(frame, config)
    fitted = preprocessor or build_preprocessor(config)
    try:
        values = fitted.fit_transform(validated)
        names = tuple(str(name) for name in fitted.get_feature_names_out())
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"cannot preprocess input data: {exc}") from exc
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise DataValidationError(
            "preprocessing produced non-finite or malformed features"
        )
    return matrix, names


def compile_customer_texts(
    frame: pd.DataFrame,
    columns: Sequence[str],
) -> tuple[str, ...]:
    """Compile each customer row into a stable embedding document."""
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise DataValidationError(f"cannot compile text; missing: {', '.join(missing)}")
    if frame.empty:
        raise DataValidationError("cannot compile text from an empty data frame")
    records = frame.loc[:, list(columns)].to_dict(orient="records")
    return tuple(
        "; ".join(
            f"{column.replace('_', ' ').title()}: {record[column]}" for column in columns
        )
        for record in records
    )

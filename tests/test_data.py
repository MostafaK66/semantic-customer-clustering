from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from semantic_customer_clustering.config import DataConfig
from semantic_customer_clustering.data import (
    build_preprocessor,
    compile_customer_texts,
    prepare_classical_features,
    read_customer_data,
    validate_customer_data,
)
from semantic_customer_clustering.errors import DataValidationError


def test_read_customer_data_passes_path_and_separator(
    customer_frame: pd.DataFrame, tmp_path: Path
) -> None:
    calls: list[tuple[object, object]] = []

    def reader(path: object, *, sep: object) -> pd.DataFrame:
        calls.append((path, sep))
        return customer_frame

    config = DataConfig(path=tmp_path / "input.csv", delimiter="|")
    result = read_customer_data(config, reader=reader)
    assert calls == [(config.path, "|")]
    assert list(result.columns) == list(config.columns)
    assert "ignored" not in result


@pytest.mark.parametrize("error", [OSError("denied"), pd.errors.ParserError("bad")])
def test_read_customer_data_wraps_reader_errors(error: Exception) -> None:
    def reader(*args: object, **kwargs: object) -> pd.DataFrame:
        del args, kwargs
        raise error

    with pytest.raises(DataValidationError, match="cannot read"):
        read_customer_data(DataConfig(), reader=reader)


def test_validate_returns_defensive_copy(customer_frame: pd.DataFrame) -> None:
    result = validate_customer_data(customer_frame, DataConfig())
    result.iloc[0, 0] = 999
    assert customer_frame.iloc[0, 0] == 21


def test_validate_rejects_empty_frame() -> None:
    with pytest.raises(DataValidationError, match="no rows"):
        validate_customer_data(pd.DataFrame(), DataConfig())


def test_validate_rejects_missing_columns(customer_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="loan"):
        validate_customer_data(customer_frame.drop(columns="loan"), DataConfig())


def test_validate_rejects_nulls(customer_frame: pd.DataFrame) -> None:
    frame = customer_frame.copy()
    frame.loc[100, "job"] = None
    with pytest.raises(DataValidationError, match="job"):
        validate_customer_data(frame, DataConfig())


def test_prepare_classical_features(customer_frame: pd.DataFrame) -> None:
    matrix, names = prepare_classical_features(customer_frame, DataConfig())
    assert matrix.shape[0] == len(customer_frame)
    assert matrix.shape[1] == len(names)
    assert np.isfinite(matrix).all()
    assert any(name.startswith("nominal") for name in names)


@pytest.mark.parametrize(
    "config",
    [
        DataConfig(columns=("age",), categorical_columns=(), ordinal_columns=()),
        DataConfig(
            columns=("job",),
            categorical_columns=("job",),
            ordinal_columns=(),
        ),
        DataConfig(
            columns=("education",),
            categorical_columns=("education",),
            ordinal_columns=("education",),
        ),
    ],
)
def test_build_preprocessor_supports_each_feature_kind(
    config: DataConfig,
) -> None:
    assert build_preprocessor(config).transformers


class FakePreprocessor:
    def __init__(self, values: object, error: Exception | None = None) -> None:
        self.values = values
        self.error = error

    def fit_transform(self, frame: pd.DataFrame) -> object:
        del frame
        if self.error:
            raise self.error
        return self.values

    def get_feature_names_out(self) -> np.ndarray:
        return np.asarray(["one", "two"])


def test_prepare_wraps_preprocessor_failure(customer_frame: pd.DataFrame) -> None:
    with pytest.raises(DataValidationError, match="cannot preprocess"):
        prepare_classical_features(
            customer_frame,
            DataConfig(),
            preprocessor=FakePreprocessor([], ValueError("broken")),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("values", [[[1, np.nan]], [1, 2]])
def test_prepare_rejects_invalid_transformed_matrix(
    customer_frame: pd.DataFrame, values: Any
) -> None:
    with pytest.raises(DataValidationError, match="malformed"):
        prepare_classical_features(
            customer_frame,
            DataConfig(),
            preprocessor=FakePreprocessor(values),  # type: ignore[arg-type]
        )


def test_compile_customer_texts_is_stable(customer_frame: pd.DataFrame) -> None:
    result = compile_customer_texts(customer_frame, ("age", "job"))
    assert result[0] == "Age: 21; Job: student"
    assert len(result) == len(customer_frame)


def test_compile_customer_texts_rejects_missing_and_empty(
    customer_frame: pd.DataFrame,
) -> None:
    with pytest.raises(DataValidationError, match="missing"):
        compile_customer_texts(customer_frame, ("unknown",))
    with pytest.raises(DataValidationError, match="empty"):
        compile_customer_texts(customer_frame.iloc[0:0], ("age",))

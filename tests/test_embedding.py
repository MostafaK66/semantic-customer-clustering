from __future__ import annotations

import builtins
from collections.abc import Sequence

import numpy as np
import pytest

from semantic_customer_clustering.config import SemanticConfig
from semantic_customer_clustering.embedding import SentenceTransformerEncoder
from semantic_customer_clustering.errors import (
    DependencyUnavailableError,
    ModelExecutionError,
)


class FakeModel:
    def __init__(self, result: object = ((1.0, 0.0), (0.0, 1.0))) -> None:
        self.result = result
        self.calls: list[tuple[Sequence[str], int, bool, bool]] = []

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> object:
        self.calls.append(
            (sentences, batch_size, show_progress_bar, normalize_embeddings)
        )
        return self.result


def test_encoder_loads_once_and_passes_configuration() -> None:
    model = FakeModel()
    names: list[str] = []

    def factory(name: str) -> FakeModel:
        names.append(name)
        return model

    config = SemanticConfig(model_name="local", batch_size=7, show_progress=False)
    encoder = SentenceTransformerEncoder(config, factory=factory)
    first = encoder.encode(("a", "b"))
    second = encoder.encode(("a", "b"))
    assert first.shape == second.shape == (2, 2)
    assert names == ["local"]
    assert model.calls[0] == (("a", "b"), 7, False, True)


def test_encoder_rejects_empty_input() -> None:
    with pytest.raises(ModelExecutionError, match="empty"):
        SentenceTransformerEncoder(
            SemanticConfig(), factory=lambda _: FakeModel()
        ).encode(())


@pytest.mark.parametrize(
    "result",
    [
        [1.0, 2.0],
        [[1.0, 2.0]],
        [[], []],
        [[1.0, np.nan], [2.0, 3.0]],
    ],
)
def test_encoder_rejects_invalid_model_output(result: object) -> None:
    encoder = SentenceTransformerEncoder(
        SemanticConfig(), factory=lambda _: FakeModel(result)
    )
    with pytest.raises(ModelExecutionError):
        encoder.encode(("a", "b"))


def test_encoder_wraps_model_execution_error() -> None:
    class BrokenModel(FakeModel):
        def encode(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise RuntimeError("device failed")

    encoder = SentenceTransformerEncoder(
        SemanticConfig(), factory=lambda _: BrokenModel()
    )
    with pytest.raises(ModelExecutionError, match="device failed"):
        encoder.encode(("a",))


def test_encoder_wraps_factory_error() -> None:
    def broken_factory(name: str) -> FakeModel:
        raise RuntimeError(name)

    encoder = SentenceTransformerEncoder(SemanticConfig(), factory=broken_factory)
    with pytest.raises(ModelExecutionError, match="cannot load"):
        encoder.encode(("a",))


def test_encoder_explains_missing_optional_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_import = builtins.__import__

    def missing(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "sentence_transformers":
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(DependencyUnavailableError, match="semantic"):
        SentenceTransformerEncoder(SemanticConfig()).encode(("a",))

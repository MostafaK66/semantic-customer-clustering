"""Injectable sentence-embedding boundary."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np

from semantic_customer_clustering.config import SemanticConfig
from semantic_customer_clustering.errors import (
    DependencyUnavailableError,
    ModelExecutionError,
)
from semantic_customer_clustering.models import FloatMatrix


class EmbeddingModel(Protocol):
    """Small subset of SentenceTransformer used by the application."""

    def encode(
        self,
        sentences: Sequence[str],
        *,
        batch_size: int,
        show_progress_bar: bool,
        normalize_embeddings: bool,
    ) -> object: ...


ModelFactory = Callable[[str], EmbeddingModel]


class SentenceTransformerEncoder:
    """Lazily construct a local model that may download weights on first use."""

    def __init__(
        self,
        config: SemanticConfig,
        *,
        factory: ModelFactory | None = None,
    ) -> None:
        self._config = config
        self._factory = factory
        self._model: EmbeddingModel | None = None

    def encode(self, sentences: Sequence[str]) -> FloatMatrix:
        """Encode texts and validate the model's matrix response."""
        if not sentences:
            raise ModelExecutionError("the embedding input is empty")
        model = self._load_model()
        try:
            raw = model.encode(
                sentences,
                batch_size=self._config.batch_size,
                show_progress_bar=self._config.show_progress,
                normalize_embeddings=True,
            )
        except Exception as exc:
            raise ModelExecutionError(f"sentence embedding failed: {exc}") from exc
        matrix = np.asarray(raw, dtype=np.float64)
        if matrix.ndim != 2 or matrix.shape[0] != len(sentences):
            raise ModelExecutionError(
                "embedding model returned a matrix with an unexpected shape"
            )
        if matrix.shape[1] == 0 or not np.isfinite(matrix).all():
            raise ModelExecutionError("embedding model returned invalid values")
        return matrix

    def _load_model(self) -> EmbeddingModel:
        if self._model is not None:
            return self._model
        factory = self._factory
        if factory is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as exc:
                raise DependencyUnavailableError(
                    "semantic clustering requires: pip install "
                    "'semantic-customer-clustering[semantic]'"
                ) from exc
            factory = SentenceTransformer
        try:
            self._model = factory(self._config.model_name)
        except Exception as exc:
            raise ModelExecutionError(
                f"cannot load embedding model {self._config.model_name!r}: {exc}"
            ) from exc
        return self._model

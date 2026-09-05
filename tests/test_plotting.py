from __future__ import annotations

import builtins
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

from semantic_customer_clustering.errors import (
    DependencyUnavailableError,
    ModelExecutionError,
)
from semantic_customer_clustering.models import ProjectionResult
from semantic_customer_clustering.plotting import save_projection_plot


class FakeAxis:
    def __init__(self) -> None:
        self.labels: list[str] = []

    def scatter(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def set_title(self, title: str) -> None:
        self.labels.append(title)

    def set_xlabel(self, label: str) -> None:
        self.labels.append(label)

    def set_ylabel(self, label: str) -> None:
        self.labels.append(label)

    def set_zlabel(self, label: str) -> None:
        self.labels.append(label)


class FakeFigure:
    def __init__(self, *, error: bool = False) -> None:
        self.axis = FakeAxis()
        self.error = error
        self.projection: object = None

    def add_subplot(self, value: int, *, projection: object) -> FakeAxis:
        del value
        self.projection = projection
        return self.axis

    def tight_layout(self) -> None:
        pass

    def savefig(self, path: Path) -> None:
        if self.error:
            raise OSError("disk full")
        path.write_text("plot", encoding="utf-8")


def install_fake_matplotlib(
    monkeypatch: pytest.MonkeyPatch, *, error: bool = False
) -> FakeFigure:
    figure = FakeFigure(error=error)
    matplotlib = types.ModuleType("matplotlib")
    pyplot = types.ModuleType("matplotlib.pyplot")
    pyplot.figure = lambda **kwargs: figure  # type: ignore[attr-defined]
    pyplot.close = lambda value: None  # type: ignore[attr-defined]
    matplotlib.pyplot = pyplot  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", pyplot)
    return figure


@pytest.mark.parametrize("dimensions", [2, 3])
def test_save_projection_plot(
    dimensions: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    figure = install_fake_matplotlib(monkeypatch)
    data = {f"component_{index + 1}": [0.0, 1.0] for index in range(dimensions)}
    data["cluster"] = [0, 1]
    projection = ProjectionResult(pd.DataFrame(data), (0, 1))
    path = save_projection_plot(projection, tmp_path / "nested" / "plot.png", "View")
    assert path.exists()
    assert figure.projection == ("3d" if dimensions == 3 else None)


def test_save_projection_plot_rejects_dimension(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_matplotlib(monkeypatch)
    projection = ProjectionResult(
        pd.DataFrame({"component_1": [0.0], "cluster": [0]}), (0,)
    )
    with pytest.raises(ModelExecutionError, match="dimensional"):
        save_projection_plot(projection, tmp_path / "plot.png", "View")


def test_save_projection_plot_wraps_write_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install_fake_matplotlib(monkeypatch, error=True)
    projection = ProjectionResult(
        pd.DataFrame({"component_1": [0.0], "component_2": [1.0], "cluster": [0]}),
        (0,),
    )
    with pytest.raises(ModelExecutionError, match="disk full"):
        save_projection_plot(projection, tmp_path / "plot.png", "View")


def test_save_projection_plot_explains_missing_dependency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def missing(
        name: str,
        globals: object = None,
        locals: object = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name.startswith("matplotlib"):
            raise ImportError(name)
        return real_import(name, globals, locals, fromlist, level)  # type: ignore[arg-type]

    monkeypatch.setattr(builtins, "__import__", missing)
    with pytest.raises(DependencyUnavailableError, match="plotting"):
        save_projection_plot(
            ProjectionResult(
                pd.DataFrame(
                    {"component_1": [0.0], "component_2": [1.0], "cluster": [0]}
                ),
                (0,),
            ),
            tmp_path / "plot.png",
            "View",
        )

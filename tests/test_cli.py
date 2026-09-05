from __future__ import annotations

import runpy
from pathlib import Path

import pytest

from semantic_customer_clustering.cli import build_parser, main
from semantic_customer_clustering.errors import DataValidationError
from semantic_customer_clustering.models import PipelineArtifacts


def artifacts(tmp_path: Path, *, tsne: bool = True) -> PipelineArtifacts:
    return PipelineArtifacts(
        tmp_path / "assignments.csv",
        tmp_path / "scores.csv",
        tmp_path / "pca.csv",
        tmp_path / "tsne.csv" if tsne else None,
    )


def test_build_parser_requires_command() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


@pytest.mark.parametrize(
    ("command", "function_name"),
    [
        ("classical", "run_classical"),
        ("semantic", "run_semantic"),
        ("mixed", "run_mixed"),
    ],
)
def test_main_dispatches_commands(
    command: str,
    function_name: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[dict[str, object]] = []

    def run(config: object, **kwargs: object) -> PipelineArtifacts:
        del config
        calls.append(kwargs)
        return artifacts(tmp_path)

    monkeypatch.setattr(f"semantic_customer_clustering.cli.{function_name}", run)
    result = main([command, "--skip-outliers", "--skip-tsne", "--plots"])
    assert result == 0
    assert calls == [{"detector": None, "include_tsne": False, "plots": True}]
    output = capsys.readouterr().out
    assert "assignments:" in output
    assert "t-SNE:" in output


def test_main_all_runs_semantic_then_classical(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def semantic(config: object, **kwargs: object) -> PipelineArtifacts:
        del config, kwargs
        calls.append("semantic")
        return artifacts(tmp_path, tsne=False)

    def classical(config: object, **kwargs: object) -> PipelineArtifacts:
        del config, kwargs
        calls.append("classical")
        return artifacts(tmp_path, tsne=False)

    monkeypatch.setattr("semantic_customer_clustering.cli.run_semantic", semantic)
    monkeypatch.setattr("semantic_customer_clustering.cli.run_classical", classical)
    assert main(["all", "--skip-outliers"]) == 0
    assert calls == ["semantic", "classical"]


def test_main_builds_default_outlier_detector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker = object()
    seen: list[object] = []
    monkeypatch.setattr(
        "semantic_customer_clustering.cli.default_detector", lambda config: marker
    )

    def run(config: object, **kwargs: object) -> PipelineArtifacts:
        del config
        seen.append(kwargs["detector"])
        return artifacts(tmp_path, tsne=False)

    monkeypatch.setattr("semantic_customer_clustering.cli.run_classical", run)
    assert main(["classical", "--skip-tsne"]) == 0
    assert seen == [marker]


def test_main_loads_config_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config_path = tmp_path / "config.toml"
    config_path.write_text('output_dir = "result"\n', encoding="utf-8")
    outputs: list[Path] = []

    def run(config: object, **kwargs: object) -> PipelineArtifacts:
        del kwargs
        outputs.append(config.output_dir)  # type: ignore[attr-defined]
        return artifacts(tmp_path, tsne=False)

    monkeypatch.setattr("semantic_customer_clustering.cli.run_classical", run)
    assert main(["--config", str(config_path), "classical", "--skip-outliers"]) == 0
    assert outputs == [tmp_path / "result"]


def test_main_reports_domain_error(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def broken(config: object, **kwargs: object) -> PipelineArtifacts:
        del config, kwargs
        raise DataValidationError("input is broken")

    monkeypatch.setattr("semantic_customer_clustering.cli.run_classical", broken)
    assert main(["classical", "--skip-outliers"]) == 2
    assert "error: input is broken" in capsys.readouterr().err


def test_main_does_not_print_absent_tsne(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        "semantic_customer_clustering.cli.run_classical",
        lambda config, **kwargs: artifacts(tmp_path, tsne=False),
    )
    assert main(["classical", "--skip-outliers"]) == 0
    assert "t-SNE:" not in capsys.readouterr().out


def test_module_entrypoint_returns_cli_status(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("semantic_customer_clustering.cli.main", lambda: 7)
    with pytest.raises(SystemExit) as raised:
        runpy.run_module("semantic_customer_clustering.__main__", run_name="__main__")
    assert raised.value.code == 7

"""Thin command-line entry point for clustering workflows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from semantic_customer_clustering.config import AppConfig
from semantic_customer_clustering.errors import ClusteringError
from semantic_customer_clustering.models import PipelineArtifacts
from semantic_customer_clustering.pipeline import (
    default_detector,
    run_classical,
    run_mixed,
    run_semantic,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser without causing process-level side effects."""
    parser = argparse.ArgumentParser(
        prog="customer-clustering",
        description="Classical, semantic, and mixed-data customer clustering",
    )
    parser.add_argument("--config", type=Path, help="TOML configuration path")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in (
        ("classical", "run transformed-feature K-means"),
        ("semantic", "run sentence-embedding K-means"),
        ("mixed", "run K-Prototypes on mixed data"),
        ("all", "run the original classical and semantic workflows"),
    ):
        child = subparsers.add_parser(command, help=help_text)
        child.add_argument(
            "--skip-outliers",
            action="store_true",
            help="do not apply optional ECOD filtering",
        )
        child.add_argument(
            "--skip-tsne",
            action="store_true",
            help="skip the comparatively expensive t-SNE projection",
        )
        child.add_argument(
            "--plots",
            action="store_true",
            help="save static PNG projection plots",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run a workflow, and translate domain errors to exit codes."""
    args = build_parser().parse_args(argv)
    try:
        config = AppConfig.from_toml(args.config) if args.config else AppConfig()
        detector = None if args.skip_outliers else default_detector(config)
        options = {
            "detector": detector,
            "include_tsne": not args.skip_tsne,
            "plots": args.plots,
        }
        artifacts: list[PipelineArtifacts]
        if args.command == "classical":
            artifacts = [run_classical(config, **options)]
        elif args.command == "semantic":
            artifacts = [run_semantic(config, **options)]
        elif args.command == "mixed":
            artifacts = [run_mixed(config, **options)]
        else:
            artifacts = [
                run_semantic(config, **options),
                run_classical(config, **options),
            ]
    except ClusteringError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for completed in artifacts:
        _print_artifacts(completed)
    return 0


def _print_artifacts(artifacts: PipelineArtifacts) -> None:
    print(f"assignments: {artifacts.assignments}")
    print(f"scores: {artifacts.scores}")
    print(f"PCA: {artifacts.pca}")
    if artifacts.tsne is not None:
        print(f"t-SNE: {artifacts.tsne}")

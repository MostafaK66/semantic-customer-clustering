# Semantic Customer Clustering

Production-oriented examples for comparing three customer-segmentation strategies:

- classical K-means over one-hot, ordinal, and power-transformed bank attributes;
- semantic K-means over locally generated SentenceTransformer embeddings;
- K-Prototypes over mixed numeric and categorical data.

The package selects a cluster count with silhouette analysis, optionally filters
outliers with ECOD, and writes aligned assignments, scores, PCA coordinates, and
t-SNE coordinates as CSV artifacts. It supports Python 3.11 and 3.12.

## Installation

The complete CLI includes optional model, mixed-data, and plotting dependencies.
The portable default outlier stage uses scikit-learn; the legacy ECOD adapter is
available separately with `.[ecod]` because PyOD may require native Numba/LLVM
build tooling on some platforms.

Linux and macOS:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
cp config.example.toml config.toml
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[all]"
Copy-Item config.example.toml config.toml
```

For library-only classical K-means, `python -m pip install -e .` is sufficient.
Contributors should install `.[dev]` or run `make install`.

## Input data

Put a semicolon-delimited CSV at `data/train_data.csv`, or change `data.path` and
`data.delimiter` in `config.toml`. By default the required columns are:

```text
age, job, marital, education, default, balance, housing, loan
```

Customer data is intentionally ignored by Git. The package rejects empty input,
missing columns, nulls, malformed model output, impossible cluster ranges, and
misaligned samples with domain-specific error messages.

## Usage

```bash
customer-clustering --config config.toml classical
customer-clustering --config config.toml semantic
customer-clustering --config config.toml mixed
customer-clustering --config config.toml all
```

Useful controls:

- `--skip-outliers` avoids the default Isolation Forest stage.
- `--skip-tsne` avoids the most expensive projection.
- `--plots` saves static PNGs instead of opening an interactive GUI.

Options follow the subcommand, for example:

```bash
customer-clustering --config config.toml semantic --skip-outliers --plots
```

The semantic workflow may download the configured model on first use. Model
construction is isolated behind `SentenceTransformerEncoder`, so local tests and
classical workflows never download model weights. `encode()` requests normalized
NumPy embeddings in line with the
[SentenceTransformers API](https://www.sbert.net/docs/package_reference/sentence_transformer/model.html).

## Outputs

Each workflow writes the following under `output_dir`:

- `<workflow>_assignments.csv`: source row index and cluster;
- `<workflow>_scores.csv`: candidate cluster counts and silhouette scores;
- `<workflow>_pca.csv`: aligned PCA coordinates and cluster;
- `<workflow>_tsne.csv`: aligned sampled t-SNE coordinates and cluster, unless skipped.

Plots are opt-in. All sampling is without replacement and uses one index selection
for both features and labels. t-SNE perplexity is capped below the sample count and
the current `max_iter` parameter is used, matching the
[scikit-learn t-SNE contract](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).

## Architecture

The `src/semantic_customer_clustering` package separates:

- immutable configuration and schema validation;
- pure text compilation and numeric preprocessing;
- clustering, outlier, embedding, and mixed-model boundaries;
- aligned PCA/t-SNE projections;
- filesystem artifacts and optional plotting;
- orchestration from the thin CLI.

The optional PyOD ECOD, SentenceTransformers, Gower, K-Modes, and Matplotlib imports
are lazy. Their adapters accept injected factories or models for deterministic
offline testing. No API key is required; the semantic model runs locally after its
weights are available.

## Development

```bash
python -m ruff check src tests
python -m pytest --cov=semantic_customer_clustering --cov-report=term-missing
python -m mypy
python -m compileall -q src tests
```

`make quality` runs the same local gates. GitHub Actions repeats them on Python 3.11
and 3.12. Unit tests must not use customer data, a network connection, a GPU, or a
real model download.

## History and license

This repository was restructured from the original `Clustering-with-LLM` work. The
two historical development branches were audited before consolidation; one was a
PyCharm starter and the other was already merged into `main`. See `NOTICE` for
attribution and `LICENSE` for the MIT terms. Third-party packages and downloaded
model weights retain their own licenses.

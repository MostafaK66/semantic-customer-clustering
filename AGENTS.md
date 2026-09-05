# Engineering contract

- Support Python 3.11 and 3.12 and retain the `src/` package layout.
- Keep configuration immutable and validate data at every public boundary.
- Keep model downloads, optional algorithms, plotting, and filesystem access injectable.
- Never require network access, a GPU, model weights, or private data in unit tests.
- Preserve row-to-label alignment when filtering, sampling, or projecting data.
- Run Ruff, strict mypy, branch-aware pytest coverage, and compile validation before merging.
- Do not commit customer data, generated outputs, caches, model files, or credentials.

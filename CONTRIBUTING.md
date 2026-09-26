# Contributing

Contributions that improve reproducibility, usability, documentation, or test coverage are welcome. For a large feature or a change to segmentation behavior, open an issue to discuss scope and validation.

## Development environment

Use a dedicated environment and work from the repository root. The package requires Python 3.10 or newer; Python 3.12 is the recommended starting point. See the [installation guide](installation.md) for environment creation.

```bash
python -m pip install -r requirements-dev.txt
python -m streamlit run app_enhanced.py
```

Create a branch for your changes. Keep algorithmic changes separate from documentation or interface changes where practical.

## Validation

```bash
python -m pytest
```

For a focused check, pass the relevant test file, such as `tests/test_segmentation.py`. Root-level demonstrations and benchmarks are not collected by default; inspect their data and hardware requirements before running them. Model inference requires separate setup and is not covered by a CPU-only smoke test.

When changing an algorithm, include a deterministic example that checks the behavior being changed. State the environment, dataset assumptions, and any checks that could not be run. A successful import or launch is not evidence of model accuracy.

## Code and documentation

- Use clear English in comments, docstrings, interface text, logs, and documentation.
- Keep reusable interface text in `locales/en_US.json`; match formatting placeholders to the calling code.
- Preserve mask semantics: binary masks, edge maps, and instance labels are not interchangeable.
- Document units, array shapes, color-channel conventions, and optional dependencies.
- Update relative documentation links when moving files.
- Keep datasets, checkpoints, outputs, secrets, and local environments out of commits.

## Pull requests and issues

Explain the concrete problem, resulting behavior, and validation. For a bug, include a minimal example, Python and dependency versions, and error output. Use public or synthetic data when sharing examples.

The project does not yet include a license grant. Resolve licensing expectations with the maintainer before submitting contributions that depend on a particular reuse license. Preserve attribution and licensing terms for third-party code and weights.

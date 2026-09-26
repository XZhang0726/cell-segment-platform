# Development Notes

Start with the [README](README.md), [installation guide](installation.md), and [contributing guide](CONTRIBUTING.md) for environment setup, entry points, and validation.

## Isolated environment

Package metadata requires Python 3.10 or newer; Python 3.12 is the recommended starting point. Use a dedicated environment and run commands from the repository root.

```bash
source .venv/bin/activate
python --version
python -m pip check
python -m streamlit run app_enhanced.py
```

On Windows PowerShell, activate `.venv` with `.\.venv\Scripts\Activate.ps1`. A Conda environment is also suitable; no particular main-environment name is required. The Windows launcher, `start-app.bat`, uses `.venv\Scripts\python.exe` when present and otherwise uses `python` from the current PATH.

## Runtime architecture

- `app_enhanced.py` is the main Streamlit application.
- `app.py` provides the optional Gradio interface; `app_streamlit.py` provides a minimal Streamlit interface.
- `src/api/segmentation.py` exposes `CellSegmenter` and `SegmentationMethod`.
- Cellpose and the SAM wrapper run in the application environment.
- The legacy `CELLSAM` API identifier refers to Meta SAM automatic mask generation. It does not identify an integration of the separate, cell-trained CellSAM model.
- CellViT runs through `src/core/segmentation/cellvit_worker.py` in the repository-local `env_cellvit` environment. The parent application remains in its main environment.
- SAM checkpoints belong in `models/sam/`; do not commit downloaded weights or virtual environments.
- `pyproject.toml` holds package metadata and tool configuration; `setup.py` is a compatibility entry point.

## GPU diagnostics

Inspect the installed PyTorch build and detected device before enabling GPU inference:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available())"
```

CUDA requires a compatible NVIDIA GPU, driver, and PyTorch build. `CUDA available: False` is expected on a Mac. The current SAM and CellViT wrappers select CPU or CUDA; they do not select Apple MPS.

`python test_gpu_cellpose.py` is an optional local diagnostic. Read the detected device and any exceptions; a completed synthetic-image check is not an accuracy evaluation or a general performance benchmark. See [installation troubleshooting](installation.md#troubleshooting).

## Validation

Install development dependencies before running the default suite:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

For focused checks:

```bash
python -m pytest tests/test_segmentation.py tests/test_metrics.py tests/test_losses.py
python test_dst_fusion.py
```

Default pytest discovery covers `tests/`. Root-level demonstrations and diagnostics can require local images, additional packages, model weights, or a GPU; inspect their inputs before running them. In particular, configure the image directory in `test_matching_performance.py` for your machine before using that benchmark.

Performance measurements should preserve the dataset, hardware, software versions, model parameters, and measurement protocol. Report untested backends or platforms explicitly.

## Environment records

Use `python -m pip list` and `python -m pip check` to inspect the active environment. Save new dependency snapshots with clear platform and version information. Review changes before replacing an existing export. The repository's Windows/CUDA snapshots are platform-specific records; see [installation.md](installation.md#development-environment-snapshots).

## Documentation conventions

Keep public prose and examples in English. Preserve established Python identifiers and serialized keys unless an API migration is intentional. Distinguish implemented behavior, verified results, and proposed features. Keep entry points and dependency guidance aligned with `pyproject.toml`, [architecture.md](docs/architecture.md), and the [development roadmap](docs/roadmap.md).

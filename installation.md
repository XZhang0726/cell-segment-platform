# Installation Guide

This guide covers the main application, optional interfaces, SAM checkpoints, and the separate CellViT worker environment. Run commands from the repository root.

## Choose an environment

The package declares Python 3.10 or newer. **Python 3.12 is recommended** for a new environment. A complete operating-system and dependency compatibility matrix has not been established, so verify the backends you intend to use.

| Backend | Environment | Additional requirements |
| --- | --- | --- |
| Traditional segmentation | Main application environment | Dependencies in `requirements.txt` |
| Cellpose | Main application environment | Cellpose and compatible PyTorch; model weights may download on first use |
| SAM automatic masks | Main application environment | `segment-anything` and a matching checkpoint in `models/sam/` |
| CellViT | Repository-local `env_cellvit` | CellViT package, compatible dependencies, and cached/downloadable weights |

The `CellSAM` / `CELLSAM` identifiers refer to Meta's Segment Anything Model (SAM) wrapper. They do not identify an integration of the separate, cell-trained CellSAM model.

Memory and storage needs depend on image dimensions, batch size, backend, and checkpoint. Begin with small representative inputs before scaling up.

## Main application environment

### 1. Create and activate an environment

With Python 3.12 selected in your terminal:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell, replace the activation command with:

```powershell
.\.venv\Scripts\Activate.ps1
```

If you prefer Conda, use a separate named environment instead:

```bash
conda create -n cell-segment python=3.12 -y
conda activate cell-segment
```

Choose one main-environment approach. Neither the application nor the Windows launcher requires a particular Conda environment name.

### 2. Install PyTorch for your machine

For CUDA acceleration, select the command for your operating system and compute platform using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing the remaining dependencies. For a CPU-first installation, the dependency installation below includes PyTorch.

On a Mac, do not install an NVIDIA CUDA wheel. `torch.cuda.is_available()` is expected to return `False` on macOS. The repository's SAM and CellViT wrappers select CPU or CUDA; selecting GPU does not enable Apple MPS in those wrappers.

On an NVIDIA machine, verify that your selected PyTorch build and driver support the GPU.

### 3. Install project dependencies

```bash
python -m pip install -r requirements.txt
python -m pip check
```

For the optional SAM backend:

```bash
python -m pip install segment-anything
```

`requirements.txt` declares dependencies but is not a fully validated, cross-platform lockfile. Backend APIs can differ between package versions. Record the resolved versions when reporting results or troubleshooting a failure.

### 4. Verify and launch

```bash
python --version
python -c "import numpy, torch, streamlit; print('NumPy:', numpy.__version__); print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import cellpose; print('Cellpose import: OK')"
python -m streamlit run app_enhanced.py
```

Begin with a traditional segmentation method to verify upload, visualization, and export before testing optional model integrations.

On Windows, `start-app.bat` provides a launcher from the repository directory. It first looks for `.venv\Scripts\python.exe`; if that file is absent, it uses `python` from the current PATH. If you use Conda, activate your environment before launching and check which interpreter the script will select.

## Alternative interfaces

The main application is `app_enhanced.py`. Two smaller interfaces are available:

```bash
# Minimal Streamlit interface
python -m streamlit run app_streamlit.py

# Optional Gradio interface
python -m pip install gradio
python app.py
```

These interfaces expose different subsets of the project. Use the main application for the full interactive workflow described in the README.

## SAM checkpoint setup

SAM runs in the main application environment. It requires a manually downloaded checkpoint with the exact filename expected by the code.

```bash
mkdir -p models/sam
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o models/sam/sam_vit_b_01ec64.pth
```

Start with `vit_b`. Other available checkpoints are:

| Model | Required filename | Download |
| --- | --- | --- |
| `vit_b` | `sam_vit_b_01ec64.pth` | [ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) |
| `vit_l` | `sam_vit_l_0b3195.pth` | [ViT-L checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth) |
| `vit_h` | `sam_vit_h_4b8939.pth` | [ViT-H checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) |

Check download sizes and available storage. A larger checkpoint is not a guarantee of better segmentation on your images. See the [SAM guide](sam-guide.md) and [manual test protocol](sam-testing.md).

## CellViT worker environment

The wrapper starts a separate Python process from `env_cellvit` at the repository root. You must create that environment locally; it is not included when you clone the repository. The main application remains in its own environment.

The adapter targets CellViT 1.0.9 and uses an isolated NumPy 1.26.4 environment as its dependency baseline. Run the following in a separate terminal with Conda available:

```bash
conda create -p ./env_cellvit python=3.12 -y
conda activate ./env_cellvit
python -m pip install cellvit==1.0.9 numpy==1.26.4
python -m pip check
python -c "import cellvit, numpy; print('CellViT import: OK'); print('NumPy:', numpy.__version__)"
conda deactivate
```

Install a compatible PyTorch build inside `env_cellvit` if the resolved dependencies did not supply the compute platform you need. The worker also imports OpenCV. Resolve conflicts within this environment and validate inference; the version baseline is not a guarantee that every operating system or GPU is supported.

The wrapper looks for:

- Linux/macOS: `env_cellvit/bin/python`
- Windows: `env_cellvit/python.exe`, then `env_cellvit/Scripts/python.exe`

Launch the main application from the terminal where its environment is active:

```bash
python -m streamlit run app_enhanced.py
```

The worker calls CellViT's model cache helper. First use may require network access, disk space, and a model download. See the [CellViT guide](cellvit-guide.md) for preprocessing limitations and the [test protocol](cellvit-testing.md).

## Development environment snapshots

`environment.yml`, `requirements_complete.txt`, and `requirements_cellpose_gpu.txt` record Windows/CUDA development configurations. The Conda export includes platform-specific packages and a local environment prefix. These files are reference snapshots, not universal installation specifications for macOS or Linux.

To reproduce a snapshot, compare your operating system, Python version, driver, and available package builds before attempting installation. For a new setup, use the steps above and record the resolved packages. Keep the CellViT worker dependencies isolated from the main environment.

## Troubleshooting

### CUDA is unavailable

On macOS or a machine without a compatible NVIDIA GPU, use CPU mode. On an NVIDIA machine, check the active environment, PyTorch build, driver, and detected device. Installing a standalone CUDA toolkit alone does not ensure that the installed PyTorch package can use the GPU.

```bash
python -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
```

### GPU memory is exhausted

Use a smaller checkpoint, lower SAM point density, smaller test images, or CPU mode. Close unrelated GPU workloads. Measure the tradeoff instead of assuming a particular speedup.

### NumPy or binary-package import errors

An error such as `numpy.core.multiarray failed to import` can indicate incompatible binary packages. Run `python -m pip check`, inspect package versions, and rebuild the affected isolated environment with compatible versions. Avoid changing the main environment to resolve a worker-only dependency conflict.

### CellViT environment or module is missing

Check the exact `env_cellvit` location and executable paths above. Test imports inside that environment. A successful import in the main environment does not prove that the worker environment is usable.

### A model does not download or load

Check the reported path, network access, available storage, and upstream model instructions. For SAM, verify the selected model and exact checkpoint filename. Do not rename a different checkpoint to satisfy a filename check.

## Upstream resources

- [Cellpose](https://github.com/MouseLand/cellpose)
- [Segment Anything](https://github.com/facebookresearch/segment-anything)
- [CellViT](https://github.com/TIO-IKIM/CellViT)
- [PyTorch installation](https://pytorch.org/get-started/locally/)

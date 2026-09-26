# Technology Stack and Architecture

This document describes the current module boundaries, execution flow, and possible extensions. Proposed layers are labeled separately from implemented components.

## Current architecture

```text
Streamlit application: app_enhanced.py
                  |
                  v
Python segmentation API: src/api/segmentation.py
                  |
     +------------+---------------------+
     |            |                     |
Traditional    Cellpose / SAM      CellViT subprocess
algorithms     main environment    env_cellvit worker
     |            |                     |
     +------------+---------------------+
                  |
                  v
Instance matching / fusion / features
                  |
                  v
Visualization / tabular machine learning / export
```

The repository also contains U-Net training and inference utilities. Model-specific dependencies and return conventions need to be considered when combining outputs.

### Main technologies

| Area | Technologies in the project |
| --- | --- |
| Language | Python 3.10 or newer; Python 3.12 recommended |
| Numerical processing | NumPy, SciPy |
| Imaging | OpenCV, scikit-image, Pillow; other format support depends on the loader |
| Deep learning | PyTorch, Cellpose, optional SAM and CellViT packages |
| Tabular analysis | Pandas, scikit-learn, optional model-specific packages |
| Visualization | Matplotlib, Plotly, Seaborn |
| Application | Streamlit |
| Configuration and logging | YAML utilities and Loguru |

Use the [installation guide](../installation.md) for environment instructions. `pyproject.toml` contains package metadata and tool configuration; `requirements.txt` declares dependencies. The Windows/CUDA environment snapshots are platform-specific reference files.

### Current source layout

```text
cell-segment-platform/
├── app_enhanced.py          # Main Streamlit application
├── app.py                   # Optional Gradio interface
├── app_streamlit.py         # Minimal Streamlit interface
├── start-app.bat            # Windows launcher
├── src/
│   ├── api/                # Shared Python segmentation interface
│   ├── core/
│   │   ├── features/       # Morphological and image-derived features
│   │   ├── fusion/         # Matching, voting, evidence, and diagnostics
│   │   ├── models/         # U-Net architecture
│   │   ├── preprocessing/  # Image preprocessing
│   │   ├── segmentation/   # Traditional and deep-learning adapters
│   │   └── utils/          # Configuration, paths, and logging
│   ├── data/               # I/O, datasets, and augmentation
│   ├── inference/          # Model predictor
│   ├── ml/                 # Tabular machine-learning modules
│   ├── training/           # Training, configuration, losses, and metrics
│   ├── analysis/           # Package namespace
│   ├── visualization/      # Package namespace
│   └── ui/                 # Package namespace
├── tests/                  # Automated tests
├── docs/                   # Design and planning documents
├── requirements.txt        # Main dependency declarations
├── pyproject.toml          # Package metadata and tool configuration
├── setup.py                # Setuptools compatibility entry point
└── README.md
```

Some namespaces are placeholders rather than fully developed service layers. Downloaded checkpoints and local environments are runtime assets and are not supplied by the source tree.

### Model boundaries

- **Cellpose and SAM:** load in the main process/environment.
- **SAM:** the `CELLSAM` enum value is retained for compatibility; it calls Meta SAM automatic mask generation.
- **CellViT:** passes an image and parameters to `cellvit_worker.py` using temporary files and runs the interpreter inside `env_cellvit`.
- **U-Net:** the predictor loads a provided checkpoint; random weights are not usable segmentation results.

The CellViT worker uses a fixed timeout and can fall back when instance postprocessing fails. See [cellvit-guide.md](../cellvit-guide.md) for operational details.

## Data flow

```text
Input image -> preprocessing -> segmentation -> optional postprocessing/fusion
     |               |               |                       |
  metadata       parameters       output mask             label map
                                                             |
                                                             v
                                              features -> analysis -> export
```

Record image scale, preprocessing, model/checkpoint, thresholds, and package versions alongside results. Different backends can return binary masks, edge images, or instance labels; make those conventions explicit before feature extraction or fusion.

## Proposed service architecture

A possible service-oriented extension separates interfaces, application services, algorithms, and storage:

1. **Interface layer:** web GUI, desktop GUI, and CLI.
2. **Service layer:** image management, segmentation jobs, and result analysis.
3. **Algorithm layer:** preprocessing, model inference, and postprocessing.
4. **Data layer:** images, model weights, annotations, and results.

The main implementation uses Streamlit and Python modules. The optional Gradio interface and minimal Streamlit interface offer additional local entry points, rather than a separate authenticated service.

### Proposed module responsibilities

| Proposed module | Responsibilities |
| --- | --- |
| Data manager | Image import/conversion, dataset indexing, annotation management, and augmentation |
| Preprocessor | Denoising, contrast enhancement, normalization, and resizing |
| Segmentation engine | Traditional methods, model adapters, inference optimization, and batches |
| Postprocessor | Contours, instance separation, result refinement, and quality checks |
| Trainer | Data loading, training loops, validation, and checkpoints |
| Analyzer | Cell counts, morphology, statistics, and reports |
| Visualizer | Overlays, interactive inspection, model comparisons, and export |
| Service API | REST endpoints, job queues, authentication, and logging |

These names are architectural responsibilities, not necessarily importable modules.

### Alternatives considered, not promised integrations

| Area | Candidate technologies | Proposed purpose |
| --- | --- | --- |
| Additional deep learning | TensorFlow/Keras | Compatibility with other model ecosystems |
| Multidimensional viewing | napari | Interactive volume inspection |
| Web service | FastAPI, React/Vue, WebSocket | Separate backend/frontend and live job progress |
| Desktop app | PyQt5/PySide6; Tkinter as an alternative | Native application |
| Storage | SQLite; PostgreSQL or MongoDB | Dataset and result indexing |
| Experiments | MLflow | Training-run tracking and model versions |
| Model export | ONNX / TorchScript | Deployment and inference optimization |
| Containers | Docker / Docker Compose | Reproducible packaging and multi-service deployment |

A separate service would need API routes and schemas, job management, deployment configuration, and dedicated tests. See the [roadmap](roadmap.md) for the proposed development phases; use the [installation guide](../installation.md) for working launch commands.

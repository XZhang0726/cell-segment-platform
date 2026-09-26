# Development Roadmap

This planning framework organizes development into twelve phases over an illustrative 28-week sequence. Week numbers express ordering and estimated scope, not calendar dates or release commitments. Some foundations already exist; assess each proposed task against the [current architecture](architecture.md) before scheduling it.

## Phase 1: Foundation, weeks 1-2

**Goal:** establish the project structure and development tools.

Planned work:

- Create the source layout and isolated development environment.
- Establish dependency management and basic data management.
- Implement image input/output for common formats.
- Create a unit-test framework.
- Configure linting, formatting, and version-control conventions.

**Deliverables:** project skeleton, image loading/saving, and development documentation.

## Phase 2: Preprocessing, weeks 3-4

**Goal:** build configurable image preprocessing.

Planned work:

- Gaussian, median, and bilateral denoising.
- Histogram equalization and CLAHE.
- Normalization and standardization.
- Resizing and cropping.
- Rotation, flipping, scaling, and elastic deformation for augmentation.
- Configuration management and before/after visualization.

**Deliverables:** preprocessing module, comparison tools, and unit tests.

## Phase 3: Traditional segmentation, weeks 5-6

**Goal:** implement interpretable baseline methods.

Planned work:

- Otsu and adaptive thresholding.
- Canny and Sobel edge detection.
- Watershed segmentation.
- Erosion, dilation, opening, and closing.
- Contour detection/analysis and touching-object separation.
- Algorithm evaluation utilities.

**Deliverables:** algorithm library, comparison reports, examples, and documentation.

## Phase 4: Deep-learning integration, weeks 7-10

**Goal:** expose multiple model backends through an inference interface.

Candidate work includes U-Net, Attention U-Net, U-Net++, Mask R-CNN, DeepLab v3+, and Cellpose; PyTorch training utilities; GPU inference; TorchScript/ONNX optimization; and model configuration management.

**Current distinction:** the repository contains U-Net, Cellpose, CellViT, and a Meta SAM wrapper. Mask R-CNN, DeepLab, and the U-Net variants remain proposed integrations.

**Proposed deliverables:** model integrations, an inference engine, and reproducible performance benchmarks.

## Phase 5: Model training, weeks 11-13

**Goal:** develop training and evaluation workflows.

Planned work:

- Dataset loaders for multiple annotation formats.
- Training loops and validation.
- Dice, focal, and combined losses.
- IoU, Dice, precision, and recall metrics.
- Learning-rate schedules, early stopping, and checkpoints.
- MLflow experiment tracking.
- Transfer learning and training configuration files.

**Deliverables:** training utilities, experiment records, and training tutorials.

## Phase 6: Postprocessing and analysis, weeks 14-15

**Goal:** turn masks into interpretable measurements.

Planned work:

- Contour extraction/refinement and instance separation.
- Cell counting and morphology: area, perimeter, circularity, major/minor axes.
- Statistical summaries and result-quality assessment.
- CSV, Excel, and JSON exports.
- Analysis-report generation.

**Deliverables:** postprocessing, analysis tools, and report templates.

## Phase 7: Batch processing, weeks 16-17

**Goal:** make repeated processing observable and efficient.

Planned work:

- A multithreaded framework and GPU batch optimization.
- Task queues, progress reporting, and logging.
- Error handling, recovery, and result caching.
- Batch configuration management.
- Stress tests and performance profiling.

**Deliverables:** batch engine, measured optimization report, and usage guide.

## Phase 8: Visualization, weeks 18-19

**Goal:** support inspection and comparison of results.

Planned work:

- Original-image and mask-overlay views.
- Interactive annotation and side-by-side comparisons.
- Bar charts, scatter plots, and heatmaps.
- Z-stack/3D visualization.
- High-resolution image and video exports.
- napari integration.

**Deliverables:** visualization utilities, an interactive viewer, and export functions. Volume viewing and napari integration remain proposed features unless separately implemented and verified.

## Phase 9: Command-line interface, week 20

**Goal:** design a dedicated command-line workflow.

Proposed commands cover preprocessing, segmentation, batch processing, training, result analysis, and configuration, with help text and examples.

**Deliverables:** CLI implementation, documentation, and example scripts. The proposal is not evidence that a packaged `cellseg` or similar command is available. Current usage begins with Streamlit or the Python API.

## Phase 10: Graphical interface, weeks 21-24

**Goal:** provide an accessible interactive application.

Potential extensions beyond the current local interfaces include:

- **Web:** UI/UX prototypes, FastAPI backend, REST API, React/Vue frontend, uploads, live progress, visualization, authentication, and permissions.
- **Desktop:** PyQt window, image browser, parameter controls, previews, result display, and executable packaging.

**Current implementation:** the main application uses Streamlit, with Gradio and minimal Streamlit alternatives. FastAPI/React and Qt remain design options.

**Proposed deliverables:** an application, user manual, and installation package.

## Phase 11: Testing and optimization, weeks 25-26

**Goal:** measure correctness, resource use, and reliability.

Planned work:

- Unit tests with a proposed coverage target above 80%.
- Integration tests and reproducible benchmarks.
- Memory profiling, leak detection, and GPU utilization checks.
- Refactoring, documentation improvements, and bug fixes.

**Deliverables:** test and optimization reports, plus a release candidate. Coverage and performance targets must be measured before being reported as achieved.

## Phase 12: Packaging and documentation, weeks 27-28

**Goal:** prepare a documented release.

Planned work:

- Docker configuration and image building.
- Deployment, API, user, and developer documentation.
- Example datasets and tutorials.
- Demonstration videos and release notes.

**Deliverables:** proposed container image, documentation, examples, and a versioned release. The plan does not establish that a package, image, or 1.0 release has been published.

## Proposed priorities

| Priority | Scope |
| --- | --- |
| P0: essential | Foundation, image I/O, at least one validated deep-learning workflow, basic inference, and CLI |
| P1: important | Preprocessing, multiple backends, batch processing, analysis, and visualization |
| P2: enhancements | GUI, training, advanced analysis, and 3D visualization |
| P3: optional | Separate web service, user management, cloud deployment, and mobile access |

Reassess these priorities against the current Streamlit-first implementation, available data, and user needs before scheduling new work.

## Illustrative milestones

| Milestone | Proposed point | Intended outcome |
| --- | --- | --- |
| M1 | End of week 4 | Foundation and preprocessing |
| M2 | End of week 10 | Deep-learning integrations |
| M3 | End of week 17 | Batch processing and analysis |
| M4 | End of week 24 | Interactive GUI |
| M5 | End of week 28 | Versioned release |

Convert future work into scoped issues with acceptance criteria, dependencies, and reproducible checks. Do not infer current completion status from elapsed time or the existence of a planning document.

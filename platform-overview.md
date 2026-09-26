# Platform Overview and Technical Notes

Cell Segment Platform is a Python research application for image segmentation, instance-mask fusion, morphological measurement, and downstream machine-learning exploration. Its primary interface is a Streamlit application, with reusable Python modules underneath.

This document distinguishes implemented components from design proposals. The presence of an algorithm in the repository does not establish accuracy, clinical suitability, production readiness, or a performance benchmark. Use the [README](README.md) for the current entry points and [installation.md](installation.md) for setup.

## 1. Processing workflow

```text
Image input and preprocessing
              |
              v
Traditional segmentation / Cellpose / CellViT / SAM / custom U-Net
              |
              v
Instance matching and optional fusion
              |
              v
Morphological features and visualization
              |
              v
Tabular analysis, machine learning, and export
```

The code is organized into the following areas:

| Area | Location | Purpose |
| --- | --- | --- |
| Interactive application | `app_enhanced.py` | Image processing and analysis workflows |
| Segmentation API | `src/api/segmentation.py` | Shared `CellSegmenter` interface |
| Segmentation implementations | `src/core/segmentation/` | Traditional algorithms and model adapters |
| Fusion | `src/core/fusion/` | Instance matching, voting, evidence combination, and diagnostics |
| Features | `src/core/features/` | Basic and advanced morphological measurements |
| Machine learning | `src/ml/` | Supervised learning, active learning, screening, clustering, and related analysis |
| Model training | `src/training/`, `src/core/models/` | Training utilities, losses, metrics, and U-Net |
| Data handling | `src/data/` | Image input/output, datasets, and augmentation |

The implementation uses NumPy, SciPy, OpenCV, scikit-image, PyTorch, scikit-learn, Pandas, Streamlit, Matplotlib, and Plotly. Optional model backends require their own packages and weights. Package metadata requires Python 3.10 or newer; Python 3.12 is recommended. See the [installation guide](installation.md) for environment setup.

## 2. Segmentation methods

| API method | Approach | Important interpretation |
| --- | --- | --- |
| `otsu` | Global thresholding by between-class variance | Produces a binary foreground mask; useful when intensity separation is clear |
| `adaptive` | Locally computed thresholds | Handles local intensity variation; parameter choice affects foreground extraction |
| `watershed` | Distance-transform watershed | Attempts to separate touching foreground objects |
| `edge_canny` | Gradient-based edge detection | Produces edges, not automatically validated cell instances |
| `cellpose` | Cellpose model adapter | Cell/nuclei segmentation depends on the chosen model, channels, and diameter |
| `cellvit` | CellViT-256 subprocess adapter | Nuclei segmentation with input resizing/padding; see its guide for limitations |
| `cellsam` | Meta SAM automatic mask generation | Legacy name for a general-purpose SAM wrapper; not the separate cell-trained CellSAM model |
| `deep_learning` | Repository U-Net predictor | Requires a suitable trained checkpoint for meaningful predictions |

### Cellpose

Cellpose uses predicted spatial flows to identify instances. The repository wrapper exposes model selection, diameter, channels, and CPU/CUDA selection through the shared API. Upstream Cellpose features, including additional parameters and 3D workflows, are not necessarily exposed by this adapter.

Inspect the selected model's output on your data and measure accuracy and runtime with a documented protocol.

### CellViT

The wrapper invokes CellViT-256 in `env_cellvit` through a subprocess. It converts the input to RGB, resizes or pads to a square target, runs inference, postprocesses the output, and restores the original output dimensions. Resizing a whole image to a small input can lose cell-scale detail.

The worker can fall back to a binary-map channel when instance postprocessing fails. Review the worker logs before relying on counts or measurements. This adapter is not a whole-slide tiling engine. See [cellvit-guide.md](cellvit-guide.md).

### SAM

The `cellsam` method uses `segment_anything.SamAutomaticMaskGenerator` with standard Meta SAM checkpoints. It generates points automatically, filters masks, and assigns non-overlapping instance labels. The current wrapper does not expose interactive box or point prompts.

A general-purpose SAM mask is not automatically a biological cell. The area filters and overlap-assignment rules also influence output. See [sam-guide.md](sam-guide.md).

### Traditional methods and U-Net

Traditional algorithms provide interpretable baselines and can assist preprocessing. Their behavior depends on illumination, contrast, cell density, and foreground polarity. Morphological cleanup and instance labeling must be considered when comparing their outputs with instance-segmentation models.

The U-Net code provides a custom training/prediction route. Without a trained checkpoint, randomly initialized weights do not produce meaningful scientific predictions.

## 3. Instance matching and fusion

Fusion explores how predictions from several methods agree or disagree. It can improve some outcomes and worsen others; compare it with each constituent model on annotated data.

### Basic strategies

For binary masks `M_i` at a pixel and model weights `w_i`, common rules include:

- **Majority vote:** retain pixels supported by the required number of masks.
- **Weighted vote:** threshold the sum of weighted model support.
- **Union:** retain any predicted foreground, often increasing coverage and false positives.
- **Intersection:** retain only shared foreground, often reducing coverage and false positives.

Exact tie handling and thresholds are defined in `src/core/fusion/fusion_engine.py`.

### Instance matching

Before fusing object shapes, the matching code associates instances using intersection over union (IoU). The repository includes baseline and optimized matchers; the optimized variant uses bounding boxes and cropped overlap calculations to avoid unnecessary full-image work.

IoU matching can be ambiguous for split and merged detections. Matching rules, overlap thresholds, and final pixel ownership affect object counts. To evaluate runtime locally, configure the input image directory in `test_matching_performance.py` and record the inputs, settings, and hardware.

### Dempster-Shafer evidence combination

The evidence model uses a frame of discernment:

```text
Theta = {Cell, Background}
```

Mass may be assigned to `Cell`, `Background`, or the whole frame `Theta`. For example, masses of 0.7, 0.2, and 0.1 encode support for cell, background, and unresolved uncertainty. Mass on `Theta` is different from assigning all evidence to a single outcome.

For two mass functions, Dempster's normalized combination for nonempty `C` is:

```text
m12(C) = sum(m1(A) * m2(B), for A intersect B = C) / (1 - K)
K      = sum(m1(A) * m2(B), for A intersect B = empty)
```

The code discounts confidence using model-reliability settings, combines evidence, computes belief/plausibility, and records conflict and uncertainty. Reliability values are user/model assumptions; they are not measured validation scores unless calibrated separately. Near-total conflict requires explicit handling rather than interpreting the result as ordinary calibrated probability.

For more than two mass functions, `combine_multiple` applies the rule sequentially and **adds** each step's conflict coefficient. The reported multi-model conflict can therefore exceed 1; it is not a normalized overall `K` and must not be read as a probability.

The fusion engine uses a default confidence of 0.8 when an input confidence map is absent. Other utilities can generate confidence surrogates from masks. These scores should not be reported as calibrated model probabilities.

### Adaptive pixel voting

For accepted matched groups, the current implementation combines evidence conflict and shape disagreement:

```text
shape_disagreement = 1 - mean(pairwise_mask_IoU)
adjusted_conflict  = 0.6 * evidence_conflict + 0.4 * shape_disagreement
```

It then selects a pixel-voting rule using fused confidence and adjusted conflict. With `n` masks in a group, the code currently uses:

| Fused confidence | Adjusted conflict | Strategy | Required pixel votes |
| --- | --- | --- | --- |
| `> 0.75` | `< 0.2` | `ULTRA_AGGRESSIVE` | At least 1 |
| `> 0.75` | `0.2` to `< 0.4` | `AGGRESSIVE` | `max(1, int(n * 0.3))` |
| `> 0.75` | `>= 0.4` | `STANDARD_HIGH_CONF` | At least `n / 2` |
| `> 0.55` to `0.75` | `< 0.3` | `RELAXED` | `max(1, int(n * 0.4))` |
| `> 0.55` to `0.75` | `0.3` to `< 0.5` | `STANDARD_MID_CONF` | At least `n / 2` |
| `> 0.55` to `0.75` | `>= 0.5` | `STRICT` | `max(1, int(n * 0.6))` |
| `<= 0.55` | `< 0.4` | `STANDARD_LOW_CONF` | At least `n / 2` |
| `<= 0.55` | `0.4` to `< 0.6` | `STRICT_LOW_CONF` | `max(1, int(n * 0.6))` |
| `<= 0.55` | `>= 0.6`, `n >= 3` | `ULTRA_STRICT` | `max(2, int(n * 0.7))` |
| `<= 0.55` | `>= 0.6`, `n < 3` | `INTERSECTION` | All masks |

The integer conversion truncates fractional thresholds; with small groups, some named strategies can produce the same effective rule. A separate minimum-contributing-model requirement is applied before this table. `handle_conflict` accepts or rejects groups using the raw `result.conflict`; pixel-voting selection and high-conflict flags use `adjusted_conflict`. These are distinct quantities, and adjusted conflict is not guaranteed to lie in [0, 1] when raw multi-model conflict exceeds 1. This is an implementation description, not proof that these thresholds are optimal.

### Refinement and diagnostics

The fusion package contains watershed-based geometric refinement and utilities for disagreement, consistency, conflict, and uncertainty maps. Refinement uses image gradients and seed regions to revise selected boundaries. Inspect the effect because refinement is not guaranteed to improve every shape.

Fusion statistics include accepted/skipped groups, per-instance evidence, conflict, uncertainty, and strategy counts. These can help prioritize manual review and diagnose model disagreement. Their interpretation depends on the matching procedure and confidence assumptions.

## 4. Morphological features

Basic and advanced feature extraction are implemented in `src/core/features/`.

| Feature group | Examples | Interpretation |
| --- | --- | --- |
| Geometry | Area, perimeter, bounding box, centroid | Size, boundary length, and location |
| Shape | Circularity, eccentricity, solidity, major/minor axes, equivalent diameter | Compactness, elongation, and deviation from a convex shape |
| Intensity | Mean, standard deviation, other regional summaries | Brightness and within-object variation |
| Shape moments | Hu-moment descriptors | Shape comparison; the current implementation uses unnormalized central moments |
| Texture | GLCM contrast, correlation, energy, homogeneity | Local intensity organization |
| Boundary descriptors | Roughness and fractal-dimension estimates | Measures of boundary complexity |

Illustrative definitions include `circularity = 4 * pi * area / perimeter**2`, `solidity = area / convex_area`, and `equivalent_diameter = sqrt(4 * area / pi)`. Pixel calibration is needed to express lengths and areas in physical units. The current Hu-moment calculation uses central rather than normalized central moments, so scale invariance should not be assumed. Segmentation errors propagate into these measurements.

These descriptors support exploratory phenotype analysis; they do not identify disease or cell type without a validated downstream model and suitable data.

## 5. Downstream machine learning

### Supervised learning

The supervised-learning module includes correlation and mutual-information selection, recursive feature elimination, tree-based importance, scaling, categorical encoding, polynomial features, model fitting, hyperparameter search, comparison, evaluation, and persistence.

Available model families include random forests, support-vector methods, linear/logistic models, gradient boosting, nearest neighbors, decision trees, AdaBoost, and extra trees. Regression utilities include linear, ridge, lasso, and elastic-net approaches. Optional algorithms depend on installed packages.

Classification reports can include accuracy, precision, recall, F1, ROC-AUC, confusion matrices, and ROC curves. Regression reports include error metrics and fit diagnostics. Fit preprocessing on training data and preserve it for inference; use held-out data to estimate generalization.

### Active learning and experiment selection

The module contains uncertainty sampling, query by committee, Gaussian-process fitting, acquisition functions, and workflow visualizations.

Common selection scores are:

```text
Least confidence: 1 - max(P(class | sample))
Margin:           P(top class) - P(second class), selecting small margins
Entropy:          -sum(P(class | sample) * log(P(class | sample)))
```

Committee disagreement and Gaussian-process acquisition functions such as expected improvement, upper confidence bound, and probability of improvement provide other selection criteria. A typical loop is initial labels, model fitting, candidate scoring, human labeling or an experiment, then refitting.

Measure label efficiency for the actual task against a defined sampling baseline and annotation budget.

### Virtual screening

The screening utilities load trained models, process candidate feature tables, compute predictions and confidence-related scores, rank/filter results, visualize distributions, and export selected candidates. Relevant functions include `screen_dataset`, `batch_screen_files`, `rank_by_prediction`, `filter_by_confidence`, and `select_top_candidates`.

A typical workflow is model training, model saving, candidate-data validation, batch prediction, ranking, review, and export. Candidate screening scores are not experimental validation or evidence of drug efficacy. Measure any reduction in experimental workload separately.

### Anomaly detection

The module includes isolation forest, local outlier factor, one-class SVM, and elliptic-envelope approaches. These can highlight unusual feature profiles or possible quality-control problems. Outliers require interpretation; an anomaly score is not a disease diagnosis.

### Clustering, dimensionality reduction, and feature analysis

The clustering module supports k-means, DBSCAN, hierarchical clustering, and Gaussian mixtures, with silhouette and Davies-Bouldin summaries and cluster-number exploration. PCA, t-SNE, and UMAP provide lower-dimensional views, while feature-analysis utilities inspect correlations and feature importance.

Cluster labels and embeddings depend on preprocessing, distance assumptions, hyperparameters, and random seeds. Apparent groups do not independently establish new biological subtypes.

## 6. Large-image processing proposal

The following architecture is a proposal for very large images. A complete terabyte-scale, distributed, 3D tiling system has not been implemented and validated in this repository.

### Overlapping tiles and global coordinates

Proposed workflow:

1. Read overlapping tiles with surrounding context.
2. Run a selected segmentation backend per tile.
3. Retain instances whose centroids fall in that tile's owned central region.
4. Project local coordinates using `global_coordinate = local_coordinate + tile_offset`.
5. Match or suppress duplicates in the overlap regions.
6. Store sparse instances and optionally fuse model outputs in global coordinates.

A coherent example uses a 512 x 512 input tile, a 400 x 400 owned central region, a 56-pixel context margin on each side, and a 400-pixel stride. The total overlap is 112 pixels. Border tiles require special handling so that image-edge cells are not discarded.

Conceptual pseudocode:

```python
from skimage.measure import regionprops

def central_instances(labels, tile_size=512, margin=56):
    """Illustrative ownership rule; image-edge handling is still required."""
    return [
        region for region in regionprops(labels)
        if margin <= region.centroid[0] < tile_size - margin
        and margin <= region.centroid[1] < tile_size - margin
    ]
```

Centroid ownership alone cannot guarantee perfect recall, complete boundaries, or deduplication: edge cases, cells larger than the context margin, differing detections, and image borders require validation.

### Sparse representation

A proposed instance record can hold an ID, global bounding box, compressed mask, confidence, tile ID, and model source:

```python
instance = {
    "id": 1024,
    "global_bbox": (100, 100, 150, 150),
    "rle_mask": "encoded mask",
    "confidence": 0.95,
    "source_tile_idx": 12,
    "model_source": "cellpose",
}
```

This is a design illustration, not the specification of an implemented export format. Memory use, indexing cost, boundary reconstruction, multi-GPU scheduling, and 3D storage need engineering and benchmarks before claims about terabyte-scale operation are appropriate.

## 7. Evaluation and future work

A reproducible evaluation should record dataset and split, annotations, image dimensions and calibration, repository revision, dependency versions, model/checkpoint, preprocessing, parameters, hardware, cold/warm timing, peak memory, and pixel/instance metrics.

Candidate extensions include:

- Explicit 3D and time-series workflows, tracking, and volume visualization.
- Additional backends such as StarDist, Omnipose, or CellSeg3D.
- Streaming analysis, online learning, and human-guided correction.
- Region-dependent reliability calibration, hierarchical evidence combination, or learned fusion.
- Validated whole-slide tiling and sparse global-instance storage.
- Distributed execution, a service API, and integration with laboratory information systems.
- Database-backed image, feature, and result provenance.

These are development directions, not completed features. See the [development roadmap](docs/roadmap.md) and [implementation plan](docs/implementation-plan.md) for proposed work and evaluation practices.

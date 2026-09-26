# SAM Segmentation Guide

The repository provides automatic object segmentation using Meta's Segment Anything Model (SAM). Its legacy API name is `CellSAM` / `SegmentationMethod.CELLSAM`. This wrapper does **not** load the separately developed, cell-trained CellSAM model.

It uses `SamAutomaticMaskGenerator` and automatically sampled points. Interactive user-supplied point, box, and mask prompts are not exposed by this wrapper.

## Setup

SAM runs in the main application environment. Follow the [installation guide](installation.md), activate your main environment, install `segment-anything`, and check imports:

```bash
python -c "import segment_anything, torch; print('SAM dependencies: OK')"
```

Download the checkpoint for the chosen model into `models/sam/`:

```bash
mkdir -p models/sam
curl -L https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -o models/sam/sam_vit_b_01ec64.pth
```

| Model | Required checkpoint |
| --- | --- |
| `vit_b` | `sam_vit_b_01ec64.pth` |
| `vit_l` | `sam_vit_l_0b3195.pth` |
| `vit_h` | `sam_vit_h_4b8939.pth` |

Additional weights are linked in the [checkpoint table](installation.md#sam-checkpoint-setup) and [upstream SAM repository](https://github.com/facebookresearch/segment-anything#model-checkpoints).

## Python API

Run this from the repository root, replacing the image path with your input:

```python
import numpy as np
from PIL import Image
from src.api.segmentation import CellSegmenter, SegmentationMethod

image = np.asarray(Image.open("path/to/image.png").convert("RGB"))
segmenter = CellSegmenter(method=SegmentationMethod.CELLSAM)
masks = segmenter.segment(
    image,
    model_type="vit_b",
    use_gpu=False,
    points_per_side=32,
)
```

The output is an integer label map with the input image's height and width. Zero denotes background; positive values identify objects. Returned IDs can have gaps, so count positive unique labels rather than using the largest ID as the object count.

### Parameters

| Parameter | Default | Meaning |
| --- | --- | --- |
| `model_type` | `"vit_b"` | SAM checkpoint architecture: `vit_b`, `vit_l`, or `vit_h` |
| `use_gpu` | `False` | Request CUDA if available; otherwise the wrapper uses CPU |
| `points_per_side` | `32` | Sampling-grid density for automatic mask generation |
| `progress_bar` | `None` | Optional Streamlit progress object |

For an NVIDIA GPU with a compatible PyTorch installation, set `use_gpu=True`. The wrapper does not select Apple MPS.

## Streamlit workflow

```bash
python -m streamlit run app_enhanced.py
```

Upload an image, select the SAM backend, choose the checkpoint and point density, then run segmentation. Inspect the overlay and cell count before exporting. The batch-processing workflow uses the same backend for multiple images.

Start with `vit_b` and `points_per_side=32`. A denser grid samples more candidate prompts and costs more computation; it does not guarantee better cell counts. Compare 16, 32, and 48 on representative annotated images.

## Implementation details and limits

- Grayscale inputs are expanded to RGB; RGBA inputs have their alpha channel removed.
- The generator expects RGB `uint8`. The wrapper converts other dtypes, but does not perform a general high-bit-depth intensity normalization. Normalize microscopy data deliberately before inference.
- Model objects are cached by architecture and device for the lifetime of the Python process. Changing either may load another model into memory.
- Current fixed generator settings include `pred_iou_thresh=0.86`, `stability_score_thresh=0.92`, `crop_n_layers=1`, `crop_n_points_downscale_factor=2`, and `min_mask_region_area=100`.
- Masks covering half or more of the image are discarded. This can also remove a legitimate large object.
- Masks are sorted by predicted IoU; each mask receives only pixels not already assigned to a higher-ranked mask. Overlaps can therefore alter shapes or eliminate an instance.
- This is general-purpose object segmentation. Suitability for a particular cell type, stain, or imaging modality requires validation.

## Troubleshooting

| Symptom | Checks and possible adjustments |
| --- | --- |
| Checkpoint not found | Match the selected architecture to the exact filename in `models/sam/` |
| CUDA out of memory | Try `vit_b`, lower point density, CPU mode, or fewer concurrent GPU workloads |
| Too few objects | Inspect contrast and object scale; compare point densities; review the fixed area and quality filters |
| Excess fragments | Inspect noise, preprocessing, point density, and overlapping-mask behavior |
| Large objects disappear | Review the filter that removes masks covering at least half the image |
| Subsequent calls seem faster | Separate model-loading time from inference when evaluating the in-process cache |

A larger checkpoint is not guaranteed to outperform a smaller one on your dataset. Use the [manual test protocol](sam-testing.md) to record comparable results.

## References

- [Segment Anything source and checkpoints](https://github.com/facebookresearch/segment-anything)
- [Segment Anything paper](https://arxiv.org/abs/2304.02643)
